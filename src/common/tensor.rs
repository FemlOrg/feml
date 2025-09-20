use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;

use crate::common::context::FemlContext;
use crate::common::context::*;
use crate::common::def::*;
use crate::common::type_traits::*;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct FemlTensor {
    pub tensor_type: TensorType,
    pub ne: [usize; 4],
    pub nb: [usize; 4],
    pub op: FemlOpType,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: Vec<Rc<FemlTensor>>,
    pub view_src: Option<Rc<FemlTensor>>,
    pub view_offs: i64,
    pub data: *mut u8,
    pub name: *const str,
    pub extra: *mut u8,
}

pub fn feml_nbytes(tensor: &FemlTensor) -> usize {
    for i in 0..FEML_MAX_DIMS {
        if tensor.ne[i] <= 0 {
            return 0;
        }
    }
    let blck_size = feml_block_size(tensor.tensor_type);
    if blck_size == 1 {
        let mut nbytes = feml_type_size(tensor.tensor_type);
        for i in 0..FEML_MAX_DIMS {
            if tensor.ne[i] > 1 {
                nbytes += ((tensor.ne[i] - 1) as usize) * tensor.nb[i];
            }
        }
        nbytes
    } else {
        let mut nbytes = (tensor.ne[0] as usize) * tensor.nb[0] / blck_size;

        for i in 1..FEML_MAX_DIMS {
            if tensor.ne[i] > 1 {
                nbytes += ((tensor.ne[i] - 1) as usize) * tensor.nb[i];
            }
        }

        nbytes
    }
}

pub fn feml_new_tensor(
    ctx: &mut FemlContext,
    tensor_type: TensorType,
    dims: usize,
    ne: &Vec<usize>,
    view_src: Option<Rc<FemlTensor>>,
    view_offs: i64,
) -> Rc<RefCell<FemlTensor>> {
    assert!(tensor_type != TensorType::TensorUnknown);
    assert!(dims > 0 && dims < FEML_MAX_DIMS);
    let mut view_src = view_src;
    let mut view_offs = view_offs;

    if let Some(ref vs) = view_src {
        if let Some(ref inner_view_src) = vs.view_src {
            view_offs += vs.view_offs;
            view_src = Some(inner_view_src.clone());
        }
    }

    let mut data_size = feml_row_size(tensor_type, ne[0]);
    for i in 1..dims {
        data_size *= ne[i];
    }

    let data = if let Some(ref vs) = view_src {
        unsafe { vs.data.offset(view_offs as isize) }
    } else {
        ptr::null_mut()
    };

    let obj_alloc_size = if view_src.is_none() && ctx.mem_buffer.is_empty() {
        data_size
    } else {
        0
    };

    let object = feml_new_object(
        ctx,
        FemlObjectType::FemlObjectTypeTensor,
        FEML_TENSOR_SIZE + obj_alloc_size,
    );
    assert!(object.is_some());
    let object_offset = object.unwrap().offset as isize;
    let result = unsafe { get_tensor(ctx, object_offset) };

    result.tensor_type = tensor_type;
    result.ne = [1, 1, 1, 1];
    result.nb = [0; 4];
    result.op = FemlOpType::FemlOpTypeUnknown;
    result.op_params = [0; 16];
    result.flags = 0;
    result.src = Vec::new();
    result.view_src = view_src;
    result.view_offs = view_offs;
    result.data = if obj_alloc_size > 0 {
        unsafe {
            &mut *(ctx
                .mem_buffer
                .as_ptr()
                .offset((object_offset + FEML_TENSOR_SIZE as isize) as isize)
                as *mut u8)
        }
    } else {
        data
    };
    result.extra = ptr::null_mut();

    for i in 0..dims {
        (*result).ne[i] = ne[i];
    }

    (*result).nb[0] = feml_type_size(tensor_type);
    (*result).nb[1] = (*result).nb[0] * ((*result).ne[0] / feml_block_size(tensor_type));
    for i in 2..FEML_MAX_DIMS {
        (*result).nb[i] = (*result).nb[i - 1] * (*result).ne[i - 1];
    }
    Rc::new(RefCell::new(result.clone()))
}

unsafe fn get_tensor_ptr(ctx: &FemlContext, offset: isize) -> *mut FemlTensor {
    unsafe { ctx.mem_buffer.as_ptr().offset(offset) as *mut FemlTensor }
}

unsafe fn get_tensor(ctx: &FemlContext, offset: isize) -> &mut FemlTensor {
    unsafe { &mut *get_tensor_ptr(ctx, offset) }
}
