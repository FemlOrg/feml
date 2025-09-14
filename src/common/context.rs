use crate::types::FemlObjectType;
use crate::utils::FEML_MEM_ALIGN;

#[derive(Debug, Clone, Copy)]
pub struct FemlObject {
   pub offset: usize,
   pub size: usize,
   pub object_type: FemlObjectType,
}

#[derive(Debug, Clone)]
pub struct FemlContext {
    pub memory_size: usize,
    pub mem_buffer: Vec<u8>,
    pub n_objects: i32,
    pub objects: Vec<FemlObject>,
}

#[derive(Debug, Clone)]
pub struct FemlInitParams {
   pub memory_size: usize,
   pub memory_buffer: Vec<u8>,
}

impl FemlContext {
    pub fn new(&mut self, mut params: FemlInitParams) -> Self {
        if params.memory_size == 0 {
            params.memory_size = FEML_MEM_ALIGN;
        }

        let memory_size = if !params.memory_buffer.is_empty() {
            params.memory_size
        } else {
            // TODO: ADD FEML_PAD
            // FEML_PAD(params.memory_size, FEML_MEM_ALIGN)
            params.memory_size
        };

        FemlContext {
            memory_size: memory_size,
            mem_buffer: params.memory_buffer,
            n_objects: 0,
            objects: Vec::new(),
        }
    }
}

pub fn feml_new_object(
    ctx: &mut FemlContext,
    object_type: FemlObjectType,
    size: usize,
) -> Option<&FemlObject> {
    let cur_end = ctx.objects.last().map_or(0, |obj| obj.offset + obj.size);

    let size_needed = (size + FEML_MEM_ALIGN - 1) & !(FEML_MEM_ALIGN - 1);

    if cur_end + size_needed > ctx.memory_size {
        eprintln!(
            "not enough space : needed {}, available{}",
            cur_end + size_needed,
            ctx.memory_size
        );
    }

    let obj_new = FemlObject {
        offset: cur_end,
        size: size_needed,
        object_type: object_type,
    };

    ctx.objects.push(obj_new);
    ctx.objects.last()
}
