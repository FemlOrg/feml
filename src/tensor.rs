use crate::data_type::{get_block_size, get_type_size, DataType, TensorOpType, TensorType};
use crate::defs::{MAX_DIMS, MAX_SRC};
use crate::error::Result;
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::TensorStorage;
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
#[allow(dead_code)]
pub(crate) struct TensorIdArray {
    arr: [TensorId; MAX_SRC],
    len: usize,
}
#[allow(dead_code)]
impl TensorIdArray {
    fn new() -> Self {
        Self { arr: [TensorId(0); MAX_SRC], len: 0 }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn push(&mut self, tensor_id: TensorId) {
        if self.len >= MAX_SRC {
            panic!("Exceeded maximum source tensors");
        }
        self.arr[self.len] = tensor_id;
        self.len += 1;
    }

    fn as_slice(&self) -> &[TensorId] {
        &self.arr[..self.len]
    }
}

pub struct TensorInner {
    pub(crate) id: TensorId,
    pub(crate) name: String,
    pub(crate) dtype: DataType,
    pub(crate) layout: Layout,
    pub(crate) self_storage: Option<TensorStorage>,
    pub(crate) extra_storage: Option<TensorStorage>,
    pub(crate) src_tensor: TensorIdArray,
    pub(crate) length: usize,
    pub(crate) tensor_type: TensorType,
    pub(crate) view_tensor: Option<Tensor>,
    pub(crate) view_offset: usize,
    pub(crate) op_type: TensorOpType,
}

#[derive(Clone)]
pub struct Tensor(pub Arc<RefCell<TensorInner>>);

impl TensorInner {
    pub(crate) fn default() -> Self {
        Self {
            id: TensorId::new(),
            name: String::new(),
            dtype: DataType::U8,
            layout: Layout::default(),
            self_storage: None,
            extra_storage: None,
            src_tensor: TensorIdArray::new(),
            length: 0,
            tensor_type: TensorType::UNKNOWN,
            view_tensor: None,
            view_offset: 0,
            op_type: TensorOpType::UNKNOWN,
        }
    }
}

impl Tensor {
    pub fn new() -> Self {
        Tensor(Arc::new(RefCell::new(TensorInner::default())))
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
        let length = shape.len();
        {
            let mut inner = self.borrow_mut();
            inner.layout.shape = shape;
            inner.length = length;
        }
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

    pub fn set_op_type(&mut self, op_type: TensorOpType) -> &mut Self {
        self.borrow_mut().op_type = op_type;
        self
    }

    pub fn get_op_type(&self) -> TensorOpType {
        self.borrow().op_type.clone()
    }

    pub fn set_tensor_type(&mut self, tensor_type: TensorType) -> &mut Self {
        self.borrow_mut().tensor_type = tensor_type;
        self
    }

    pub fn get_tensor_type(&self) -> TensorType {
        self.borrow().tensor_type.clone()
    }

    pub fn get_view_offset(&self) -> usize {
        self.borrow().view_offset
    }

    pub fn get_data(&self) -> Result<*mut u8> {
        todo!("get_data");
    }

    pub fn nbytes(&self) -> usize {
        self.borrow().layout.nbytes(self.get_dtype())
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl std::ops::Deref for Tensor {
    type Target = RefCell<TensorInner>;

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
        assert_eq!(tensor.borrow().view_offset, 0);
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
        let tensor = TensorInner::default();
        assert!(tensor.self_storage.is_none());
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
        let tensor = Tensor(Arc::new(RefCell::new(TensorInner::default())));
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
        let mut tensor = TensorInner::default();

        assert_eq!(tensor.view_offset, 0);

        tensor.view_offset = 100;
        assert_eq!(tensor.view_offset, 100);
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
