use crate::context::Context;
use crate::data_type::{DataType, TensorOpType, TensorType};
use crate::defs::{MAX_DIMS, MAX_SRC};
use crate::error::{Error, ErrorKind, Result};
use crate::layout::Layout;
use crate::memory_manager::MemoryBlock;
use crate::shape::Shape;
use std::cell::RefCell;
use std::sync::Arc;
/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    pub fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

struct TensorIdArray {
    arr: [TensorId; MAX_SRC],
    len: usize,
}

impl TensorIdArray {
    pub(crate) fn new() -> Self {
        Self { arr: [TensorId(0); MAX_SRC], len: 0 }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn push(&mut self, tensor_id: TensorId) {
        if self.len >= MAX_SRC {
            panic!("Exceeded maximum source tensors");
        }
        self.arr[self.len] = tensor_id;
        self.len += 1;
    }

    pub(crate) fn as_slice(&self) -> &[TensorId] {
        &self.arr[..self.len]
    }
}

pub struct Tensor_ {
    pub id: TensorId,
    pub name: String,
    pub dtype: DataType,
    pub layout: Layout,
    pub storage: Option<Arc<MemoryBlock>>,
    pub src_tensor: TensorIdArray,
    pub length: usize,
    pub tensor_type: TensorType,
    pub view_offs: usize,
    pub op_type: TensorOpType,
    context: Option<Context>,
}

#[derive(Clone)]
pub struct Tensor(pub Arc<RefCell<Tensor_>>);

impl Tensor_ {
    pub(crate) fn default() -> Self {
        Self {
            id: TensorId::new(),
            name: String::new(),
            dtype: DataType::U8,
            layout: Layout::default(),
            storage: None,
            src_tensor: TensorIdArray::new(),
            length: 0,
            tensor_type: TensorType::UNKNOWN,
            view_offs: 0,
            op_type: TensorOpType::UNKNOWN,
            context: None,
        }
    }
}

impl Tensor {
    pub fn new() -> Self {
        Tensor(Arc::new(RefCell::new(Tensor_::default())))
    }

    fn mul_impl(&self, other: &Tensor, inplace: bool) -> Result<Tensor> {
        let mut result = if inplace {
            self.borrow().context.clone().unwrap().new_tensor_view(self.clone())
        } else {
            self.borrow().context.clone().unwrap().dup_tensor(self.clone())
        };

        match &mut result {
            Ok(res) => {
                res.set_src_tensor(self.get_tensor_id());
                res.set_src_tensor(other.get_tensor_id());
            }
            Err(e) => {
                eprintln!("{}", e);
            }
        }

        result
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.mul_impl(other, false)
    }

    pub fn mul_inplace(&mut self, other: &Tensor) -> Result<Tensor> {
        self.mul_impl(other, true)
    }

    pub fn set_tensor_id(&mut self, id: TensorId) -> &mut Self {
        self.borrow_mut().id = id;
        self
    }

    pub fn get_tensor_id(&self) -> TensorId {
        self.borrow().id
    }

    pub fn set_src_tensor(&mut self, src_tensor_id: TensorId) -> &mut Self {
        self.borrow_mut().src_tensor.push(src_tensor_id);
        self
    }

    pub fn get_src_tensor(&self) -> Vec<TensorId> {
        self.borrow().src_tensor.as_slice().to_vec()
    }

    pub fn set_data_type(&mut self, dtype: DataType) -> &mut Self {
        self.borrow_mut().dtype = dtype;
        self
    }

    pub fn set_shape(&mut self, shape: Shape) -> &mut Self {
        self.borrow_mut().layout.shape = shape;
        self
    }

    pub fn get_shape(&self) -> Shape {
        self.borrow().layout.shape.clone()
    }

    pub fn set_length(&mut self, length: usize) -> &mut Self {
        self.borrow_mut().length = length;
        self
    }

    pub fn get_length(&self) -> usize {
        self.borrow().length
    }

    pub fn get_dtype(&self) -> DataType {
        self.borrow().dtype.clone()
    }

    pub fn set_name(&mut self, name: String) -> &mut Self {
        self.borrow_mut().name = name;
        self
    }

    pub fn get_name(&self) -> String {
        self.borrow().name.clone()
    }

    pub fn set_data(&mut self, data: Arc<MemoryBlock>) -> &mut Self {
        self.borrow_mut().storage = Some(data);
        self
    }

    pub fn set_op_type(&mut self, op_type: TensorOpType) -> &mut Self {
        self.borrow_mut().op_type = op_type;
        self
    }

    pub fn get_op_type(&self) -> TensorOpType {
        self.borrow().op_type.clone()
    }

    pub fn set_context(&mut self, context: Context) -> &mut Self {
        self.borrow_mut().context = Some(context);
        self
    }

    pub fn get_tensor_type(&self) -> TensorType {
        self.borrow().tensor_type.clone()
    }

    pub fn get_view_offset(&self) -> usize {
        self.borrow().view_offs
    }

    pub fn get_data(&self) -> Result<*mut u8> {
        // self.storage
        //     .as_ref()
        //     .ok_or("Tensor storage is None".to_string())
        todo!("get_data");
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl std::ops::Deref for Tensor {
    type Target = RefCell<Tensor_>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_id_uniqueness() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        let id3 = TensorId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_tensor_id_copy_clone() {
        let id1 = TensorId::new();
        let id2 = id1;
        let id3 = id1.clone();

        assert_eq!(id1, id2);
        assert_eq!(id1, id3);
    }

    #[test]
    fn test_tensor_id_traits() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let _ = format!("{:?}", id1);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(id1);
        set.insert(id2);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_tensor_default() {
        let tensor = Tensor::new();

        assert_eq!(tensor.get_name(), String::new());
        assert_eq!(tensor.get_dtype(), DataType::U8);
        assert_eq!(tensor.get_length(), 0);
        assert_eq!(tensor.get_tensor_type(), TensorType::UNKNOWN);
        assert_eq!(tensor.get_op_type(), TensorOpType::UNKNOWN);
        assert_eq!(tensor.borrow().view_offs, 0);
    }

    #[test]
    fn test_tensor_setters() {
        let mut tensor = Tensor::new();

        let name = "test_tensor".to_string();
        tensor.set_name(name.clone());
        assert_eq!(tensor.get_name(), name);

        for dtype in [
            DataType::U8,
            DataType::U32,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::F16,
            DataType::F32,
            DataType::F64,
        ] {
            tensor.set_data_type(dtype);
            assert_eq!(tensor.get_dtype(), dtype);
        }

        let shape = Shape([2, 3, 4, 5]);
        tensor.set_shape(shape);
        assert_eq!(tensor.get_shape(), shape);

        tensor.set_length(100);
        assert_eq!(tensor.get_length(), 100);

        for op_type in
            [TensorOpType::UNKNOWN, TensorOpType::TensorOpView, TensorOpType::TensorOpMul]
        {
            tensor.set_op_type(op_type);
            assert_eq!(tensor.get_op_type(), op_type);
        }

        for tensor_type in [TensorType::UNKNOWN, TensorType::InputParam, TensorType::OutputParam] {
            tensor.borrow_mut().tensor_type = tensor_type;
            assert_eq!(tensor.borrow().tensor_type, tensor_type);
        }
    }

    #[test]
    fn test_setter_chaining() {
        let mut tensor = Tensor::new();

        let _ = tensor
            .set_name("chained".to_string())
            .set_data_type(DataType::F32)
            .set_shape(Shape([1, 2, 3, 4]))
            .set_length(42);

        assert_eq!(tensor.get_name(), "chained".to_string());
        assert_eq!(tensor.get_dtype(), DataType::F32);
        assert_eq!(tensor.get_shape(), Shape([1, 2, 3, 4]));
        assert_eq!(tensor.get_length(), 42);
    }

    #[test]
    fn test_tensor_data() {
        let tensor = Tensor_::default();
        assert!(tensor.storage.is_none());
    }

    #[test]
    fn test_tensor_clone() {
        let mut t1 = Tensor::new();
        t1.set_name("original".to_string());

        let mut t2 = t1.clone();

        assert_eq!(t1.get_name(), t2.get_name());

        t2.set_name("copy".to_string());
        assert_eq!(t1.get_name(), "copy".to_string());
        assert_eq!(t2.get_name(), "copy".to_string());
    }

    #[test]
    fn test_tensor_as_ref() {
        let tensor = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let refed: &Tensor = tensor.as_ref();

        assert_eq!(std::ptr::eq(refed, &tensor), true);
    }

    #[test]
    fn test_shape_equality() {
        let s1 = Shape([1, 2, 3, 4]);
        let s2 = Shape([1, 2, 3, 4]);
        let s3 = Shape([2, 3, 4, 5]);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_shape_debug() {
        let shape = Shape([1, 2, 3, 4]);
        let debug_str = format!("{:?}", shape);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
        assert!(debug_str.contains("4"));
    }

    #[test]
    fn test_shape_default() {
        let shape = Shape::default();
        assert_eq!(shape, Shape([0, 0, 0, 0]));
    }

    #[test]
    fn test_datatype_copy_clone() {
        let dt1 = DataType::F32;
        let dt2 = dt1;
        let dt3 = dt1.clone();

        assert_eq!(dt1, dt2);
        assert_eq!(dt1, dt3);
    }

    #[test]
    fn test_tensortype_copy_clone() {
        let tt1 = TensorType::InputParam;
        let tt2 = tt1;
        let tt3 = tt1.clone();

        assert_eq!(tt1, tt2);
        assert_eq!(tt1, tt3);
    }

    #[test]
    fn test_tensoroptype_copy_clone() {
        let to1 = TensorOpType::TensorOpMul;
        let to2 = to1;
        let to3 = to1.clone();

        assert_eq!(to1, to2);
        assert_eq!(to1, to3);
    }

    #[test]
    fn test_all_datatypes() {
        let types = [
            DataType::U8,
            DataType::U32,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::F16,
            DataType::F32,
            DataType::F64,
        ];

        for i in 0..types.len() {
            for j in i + 1..types.len() {
                assert_ne!(types[i], types[j]);
            }
        }

        for dt in types.iter() {
            let _ = format!("{:?}", dt);
        }
    }

    #[test]
    fn test_get_size() {
        use crate::data_type::get_type_size;

        assert_eq!(get_type_size(DataType::U8), 1);
        assert_eq!(get_type_size(DataType::U32), 4);
        assert_eq!(get_type_size(DataType::I16), 2);
        assert_eq!(get_type_size(DataType::I32), 4);
        assert_eq!(get_type_size(DataType::I64), 8);
        assert_eq!(get_type_size(DataType::F16), 2);
        assert_eq!(get_type_size(DataType::F32), 4);
        assert_eq!(get_type_size(DataType::F64), 8);
    }

    #[test]
    fn test_tensor_id_field() {
        let mut tensor = Tensor::new();
        let new_id = TensorId::new();

        tensor.set_tensor_id(new_id);
        assert_eq!(tensor.get_tensor_id(), new_id);
    }

    #[test]
    fn test_tensor_view_offs() {
        let mut tensor = Tensor_::default();

        assert_eq!(tensor.view_offs, 0);

        tensor.view_offs = 100;
        assert_eq!(tensor.view_offs, 100);
    }

    #[test]
    fn test_tensor_with_different_dtypes() {
        let dtypes = [
            DataType::U8,
            DataType::U32,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::F16,
            DataType::F32,
            DataType::F64,
        ];

        for dtype in dtypes {
            let mut tensor = Tensor::new();
            tensor.set_data_type(dtype);
            assert_eq!(tensor.get_dtype(), dtype);
        }
    }
}
