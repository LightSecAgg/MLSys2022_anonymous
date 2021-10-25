import logging
import os
import sys

import numpy as np

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.client.client_manager import ClientManager
from .message_define import MyMessage
from .utils import transform_list_to_tensor, transform_tensor_to_finite
from .utils import model_dimension
from .mpc_function import model_masking, mask_encoding, compute_aggregate_encoded_mask


class LightSecAggClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.local_mask = None
        self.encoded_mask_dict = dict()
        self.flag_encoded_mask_dict = dict()
        self.worker_num = size - 1
        self.dimensions = []
        self.total_dimension = None
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False

        # new added parameters in main file
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT, self.handle_message_receive_encoded_mask_from_server
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.handle_message_receive_active_from_server
        )

    def handle_message_init(self, msg_params):
        logging.info("client %d handle_message_init from server." % self.get_sender_id())
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.dimensions, self.total_dimension = model_dimension(global_model_params)
        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__offline()

    def handle_message_receive_encoded_mask_from_server(self, msg_params):
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        logging.info(
            "Client %d receive encoded_mask = %s from Client %d" % (self.get_sender_id(), encoded_mask, client_id)
        )
        self.add_encoded_mask(client_id - 1, encoded_mask)
        b_all_received = self.check_whether_all_encoded_mask_receive()
        if b_all_received:
            # Start the local training if receive all the encoded masks
            self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("client %d handle_message_receive_model_from_server." % self.get_sender_id())
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1

        self.__offline()

        if self.round_idx == self.num_rounds - 1:
            logging.info("this client has finished the training!")

    def handle_message_receive_active_from_server(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # Receive the set of active client id in first round
        active_clients_first_round = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS)
        logging.info(
            "Client %d receive active_clients in the first round = %s"
            % (self.get_sender_id(), active_clients_first_round)
        )

        # Compute the aggregate of encoded masks for the active clients
        p = self.prime_number
        aggregate_encoded_mask = compute_aggregate_encoded_mask(self.encoded_mask_dict, p, active_clients_first_round)

        # Send the aggregate of encoded mask to server
        self.send_aggregate_encoded_mask_to_server(0, aggregate_encoded_mask)

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_encoded_mask_to_server(self, receive_id, encoded_mask):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, receive_id)
        self.send_message(message)

    def send_aggregate_encoded_mask_to_server(self, receive_id, aggregate_encoded_mask):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK, aggregate_encoded_mask)
        self.send_message(message)

    def add_encoded_mask(self, index, encoded_mask):
        self.encoded_mask_dict[index] = encoded_mask
        self.flag_encoded_mask_dict[index] = True

    def check_whether_all_encoded_mask_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_encoded_mask_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False
        return True

    def encoded_mask_sharing(self, encoded_mask_set):
        for receive_id in range(1, self.size):
            encoded_mask = encoded_mask_set[receive_id - 1]
            if receive_id != self.get_sender_id():
                self.send_encoded_mask_to_server(receive_id, encoded_mask)
            else:
                self.encoded_mask_dict[receive_id - 1] = encoded_mask
                self.flag_encoded_mask_dict[receive_id - 1] = True

    def __offline(self):
        # Encoding the local generated mask
        logging.info("#######Client %d offline encoding round_id = %d######" % (self.get_sender_id(), self.round_idx))

        # encoded_mask_set = self.mask_encoding()
        d = self.total_dimension
        N = self.size - 1
        U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number
        logging.debug("d = {}, N = {}, U = {}, T = {}, p = {}".format(d, N, U, T, p))
        self.local_mask = np.random.randint(p, size=(d, 1))
        encoded_mask_set = mask_encoding(d, N, U, T, p, self.local_mask)

        # Send the encoded masks to other clients (via server)
        self.encoded_mask_sharing(encoded_mask_set)

    def __train(self):
        logging.info("Client %d #######training########### round_id = %d" % (self.get_sender_id(), self.round_idx))
        weights, local_sample_num = self.trainer.train(self.round_idx)

        # Convert the model from real to finite
        p = self.prime_number
        q_bits = self.precision_parameter
        weights_finite = transform_tensor_to_finite(weights, p, q_bits)

        # Mask the local model
        masked_weights = model_masking(weights_finite, self.dimensions, self.local_mask, self.prime_number)

        self.send_model_to_server(0, masked_weights, local_sample_num)
