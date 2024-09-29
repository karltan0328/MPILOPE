def get_sam_info(type:str='h') -> tuple:
    if type == 'h':
        sam_checkpoint = './weights/sam_vit_h_4b8939.pth'
        model_type = 'vit_h'
    elif type == 'l':
        sam_checkpoint = './weights/sam_vit_l_0b3195.pth'
        model_type = 'vit_l'
    elif type == 'b':
        sam_checkpoint = './weights/sam_vit_b_01ec64.pth'
        model_type = 'vit_b'
    else:
        raise NotImplementedError
    return model_type, sam_checkpoint
