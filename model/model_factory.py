from model.axial_attention_model import AxialAttentionModel, PositionalAxialAttention
from model.util.axial_attention import AxialPositionBlock, AxialPositionGateBlock, AxialWithoutPositionBlock
from model.util.axial_attention_3d import Axial3DPositionBlock, Axial3DPositionGateBlock, Axial3DWithoutPositionBlock


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

    if model_name == "AxialAttentionWithoutPosition_3D":
        return AxialAttentionModel(Axial3DWithoutPositionBlock, [1, 2, 4, 1],
                                   num_classes=num_classes,
                                   s=0.125)

    if model_name == "AxialAttentionPosition_3D":
        return PositionalAxialAttention(Axial3DPositionBlock, [1, 2, 4, 1],
                                        num_classes=num_classes,
                                        s=0.125)

    if model_name == "AxialAttentionPositionGate_3d":
        return PositionalAxialAttention(Axial3DPositionGateBlock, [1, 2, 4, 1],
                                        num_classes=num_classes,
                                        s=0.125)
