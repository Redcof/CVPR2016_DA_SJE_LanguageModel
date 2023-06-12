import torch

def combine_all_parameters(*networks):
    parameters = []
    gradParameters = []

    for network in networks:
        if hasattr(network, 'weights'):
            parameters.append(network.weights)
            gradParameters.append(network.grads)
        else:
            net_params = list(network.parameters())
            if net_params:
                parameters.extend(net_params)
                gradParameters.extend(network.grads for _ in net_params)

    def storage_in_set(_set, storage):
        storage_and_offset = _set.get(torch.pointer(storage))
        if storage_and_offset is None:
            return None
        _, offset = storage_and_offset
        return offset

    def flatten(parameters):
        if not parameters or len(parameters) == 0:
            return torch.Tensor()

        Tensor = parameters[0].new

        storages = {}
        nParameters = 0

        for parameter in parameters:
            storage = parameter.storage()
            if storage_in_set(storages, storage) is None:
                storages[torch.pointer(storage)] = (storage, nParameters)
                nParameters += storage.size()

        flatParameters = Tensor(nParameters).fill_(1)
        flatStorage = flatParameters.storage()

        for parameter in parameters:
            storage_offset = storage_in_set(storages, parameter.storage())
            parameter.set_(flatStorage,
                           storage_offset + parameter.storage_offset(),
                           parameter.size(),
                           parameter.stride())
            parameter.zero_()

        maskParameters = flatParameters.float().clone()
        cumSumOfHoles = flatParameters.float().cumsum(0)
        nUsedParameters = nParameters - cumSumOfHoles[-1]
        flatUsedParameters = Tensor(nUsedParameters)
        flatUsedStorage = flatUsedParameters.storage()

        for parameter in parameters:
            offset = cumSumOfHoles[parameter.storage_offset()]
            parameter.set_(flatUsedStorage,
                           parameter.storage_offset() - offset,
                           parameter.size(),
                           parameter.stride())

        for storage, offset in storages.values():
            flatParameters[v+1:v+storage.size()+1] = Tensor().set_(storage)

        if cumSumOfHoles.sum() == 0:
            flatUsedParameters.copy_(flatParameters)
        else:
            counter = 0
            for k in range(flatParameters.numel()):
                if maskParameters[k] == 0:
                    counter += 1
                    flatUsedParameters[counter-1] = flatParameters[counter+cumSumOfHoles[k]-1]

            assert counter == nUsedParameters

        return flatUsedParameters

    flatParameters = flatten(parameters)
    flatGradParameters = flatten(gradParameters)

    return flatParameters, flatGradParameters

def clone_many_times(net, T):
    clones = []

    params, gradParams = None, None
    if hasattr(net, 'parameters'):
        params, gradParams = net.parameters()
        if params is None:
            params = []

    paramsNoGrad = None
    if hasattr(net, 'parametersNoGrad'):
        paramsNoGrad = net.parametersNoGrad()

    mem = torch.MemoryFile(torch.ByteStorage()).writeObject(net)

    for t in range(1, T+1):
        reader = torch.MemoryFile(mem.storage(), "r").readObject()
        clone = reader
        reader.close()

        if hasattr(net, 'parameters'):
            cloneParams, cloneGradParams = clone.parameters()
            cloneParamsNoGrad = None
            for i in range(len(params)):
                cloneParams[i].set_(params[i])
                cloneGradParams[i].set_(gradParams[i])
            if paramsNoGrad:
                cloneParamsNoGrad = clone.parametersNoGrad()
                for i in range(len(paramsNoGrad)):
                    cloneParamsNoGrad[i].set_(paramsNoGrad[i])

        clones.append(clone)
        torch.cuda.empty_cache()

    mem.close()
    return clones

def split(inputstr, sep=None):
    if sep is None:
        sep = "%s"
    t = []
    for str in inputstr.split(sep):
        t.append(str)
    return t

def trim(s):
    return s.strip()

model_utils = {
    'clone_many_times': clone_many_times,
    'split': split,
    'trim': trim
}

return model_utils
