import torch

def combine_masks_with_batch(masks, n_obj, th=0.5, return_as_onehot = False):
    """ Combine mask for different objects.

    Different methods are the following:

    * `max_per_pixel`: Computes the final mask taking the pixel with the highest
                       probability for every object.

    # Arguments
        masks: Tensor with shape[B, nobj, H, W]. H, W on batches must be same
        method: String. Method that specifies how the masks are fused.

    # Returns
        [B, 1, H, W]
    """

    # masks : B, nobj, h, w
    # output : h,w
    marker = torch.argmax(masks, dim=1, keepdim=True) #
    if not return_as_onehot:
        out_mask = torch.unsqueeze(torch.zeros_like(masks)[:,0],1) #[B, 1, H, W]
        for obj_id in range(n_obj):
            try :tmp_mask = (marker == obj_id) * (masks[:,obj_id].unsqueeze(1) > th)
            except: raise NotImplementedError
            out_mask[tmp_mask] = obj_id + 1 # [B, 1, H, W]

    if return_as_onehot:
        out_mask = torch.zeros_like(masks) # [B, nobj, H, W]
        for obj_id in range(n_obj):
            try :tmp_mask = (marker == obj_id) * (masks[:,obj_id].unsqueeze(1) > th)
            except: raise NotImplementedError
            out_mask[:, obj_id] = tmp_mask[:,0].type(torch.cuda.FloatTensor)

    return out_mask
