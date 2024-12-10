from yolo_cam.base_cam import BaseCAM
from yolo_cam.utils.svd_on_activations import get_2d_projection


class GradMap(BaseCAM):
    def __init__(self, model, target_layers, task: str = 'od',
                 reshape_transform=None):
        super(GradMap, self).__init__(model,
                                      target_layers,
                                      task,
                                      reshape_transform,
                                      uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
