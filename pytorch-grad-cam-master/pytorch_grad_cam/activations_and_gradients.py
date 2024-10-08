'''class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
'''

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.layerActivation = False
        self.layerGradient = False
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        self.target_layers = target_layers

        # Register hooks for each target layer
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append( 
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if (str(module) == "LayerNorm((1024,), eps=1e-05, elementwise_affine=True)" and 
            self.layerActivation == False):
            self.layerActivation = True
            #print(f"Activation Hook Triggered for: {module}")  # Debugging line

            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
                self.activations.append(activation.cpu().detach())

        if (str(module) == "LayerNorm((192,), eps=1e-06, elementwise_affine=True)" and 
            self.layerActivation == False):
            self.layerActivation = True
            #print(f"Activation Hook Triggered for: {module}")  # Debugging line

            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
                self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        if (str(module) == "LayerNorm((1024,), eps=1e-05, elementwise_affine=True)" and
            self.layerGradient == False):
            self.layerGradient = True
            #print(f"Gradient Hook Triggered for: {module}")  # Debugging line
            # Assuming you want to store only the first gradient (index 0) in grad_output
            grad = grad_output#[0]
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        if (str(module) == "LayerNorm((192,), eps=1e-06, elementwise_affine=True)" and 
            self.layerGradient == False):
            self.layerGradient = True
            #print(f"Gradient Hook Triggered for: {module}")  # Debugging line
            # Assuming you want to store only the first gradient (index 0) in grad_output
            grad = grad_output#[0]
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        output = self.model(x)
        return output

    def release(self):
        for handle in self.handles:
            handle.remove()