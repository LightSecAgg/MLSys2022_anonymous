import logging
from time import sleep

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from .message_define import MyMessage
from .utils import transform_tensor_to_list


class LightSecAggServerManager(ServerManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

        self.active_clients_first_round = []
        self.active_clients_second_round = []

        ### new added parameters in main file ###
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
        )
        global_model_params = self.aggregator.get_global_model_params()
        self.aggregator.get_model_dimension(global_model_params)

        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.handle_message_receive_encoded_mask_from_client
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.handle_message_receive_aggregate_encoded_mask_from_client
        )

    def handle_message_receive_encoded_mask_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        self.send_message_encoded_mask_to_client(sender_id, receive_id, encoded_mask)

    def handle_message_receive_model_from_client(self, msg_params):
        # Receive the masked models from clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("Server: model_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)
        if b_all_received:
            # Specify the active clients for the first round and inform them
            for receiver_id in range(1, self.size):
                self.send_message_to_active_client(receiver_id, self.active_clients_first_round)

    def handle_message_receive_aggregate_encoded_mask_from_client(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        aggregate_encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK)
        self.aggregator.add_local_aggregate_encoded_mask(sender_id - 1, aggregate_encoded_mask)
        logging.info(
            "Server handle_message_receive_aggregate_mask = %d from_client =  %d"
            % (len(aggregate_encoded_mask), sender_id)
        )
        # Active clients for the second round
        self.active_clients_second_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_aggregate_encoded_mask_receive()
        logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if b_all_received:
            # Secure Model Aggregation
            global_model_params = self.aggregator.aggregate_model_reconstruction(
                self.active_clients_first_round, self.active_clients_second_round
            )
            # evaluation
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            self.active_clients_first_round = []
            self.active_clients_second_round = []

            if self.round_idx == self.round_num:
                logging.info("=================TRAINING IS FINISHED!=============")
                sleep(3)
                self.finish()
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1]
                )

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_encoded_mask_to_client(self, sender_id, receive_id, encoded_mask):
        message = Message(MyMessage.MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, sender_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("Server send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_to_active_client(self, receive_id, active_clients):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.send_message(message)
