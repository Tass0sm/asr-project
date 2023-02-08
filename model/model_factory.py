from model.axial_attention_model import AxialAttentionModel, PositionalAxialAttention
from model.util.axial_attention import AxialPositionBlock, AxialPositionGateBlock, AxialWithoutPositionBlock


def get_model(model_name, num_classes=50):
    if model_name == "AxialAttentionWithoutPosition":
        return AxialAttentionModel(AxialWithoutPositionBlock, [1, 2, 4, 1],
                                   num_classes=num_classes,
                                   s=0.125)

    if model_name == "AxialAttentionPosition":
        return PositionalAxialAttention(AxialPositionBlock, [1, 2, 4, 1],
                                        num_classes=num_classes,
                                        s=0.125)

    if model_name == "AxialAttentionPositionGate":
        return PositionalAxialAttention(AxialPositionGateBlock, [1, 2, 4, 1],
                                        num_classes=num_classes,
                                        s=0.125)
