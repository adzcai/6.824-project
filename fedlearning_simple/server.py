import rpyc

import numpy as np
import torch
from utils import Net


class FederatedLearningService(rpyc.Service):
    def __init__(self):
        super(FederatedLearningService, self).__init__()
        self.net = Net()

    def exposed_send_gradient(self, param_gradients):
        with torch.no_grad():
            for param, grad in zip(self.net.parameters(), param_gradients):
                param -= torch.as_tensor(np.array(grad)).clip(-0.1, 0.1) * 0.001

    def exposed_get_model_params(self):
        params = list(map(lambda x: x.data.numpy(), self.net.parameters()))
        return params


if __name__ == "__main__":
    service = FederatedLearningService()
    server = rpyc.utils.server.ThreadedServer(
        service,
        hostname="localhost",
        port=12345,
        protocol_config={
            "allow_pickle": True,
        })
    server.start()
