use crate::context::Context;
use crate::data_type::{DataType, TensorOpType, TensorType};
use crate::memory_manager::MemoryBlock;
use crate::shape::Shape;
use crate::layout::Layout;
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
}
pub struct Tensor_ {
    pub id: TensorId,
    pub name: String,
    pub dtype: DataType,
    pub layout: Layout,
    pub storage: Option<Arc<MemoryBlock>>,
    pub src_tensor: Vec<Tensor>,
    pub length: usize,
    pub tensor_type: TensorType,
    pub view_offs: usize,
    pub op_type: TensorOpType,
}

#[derive(Clone)]
pub struct Tensor(pub Arc<RefCell<Tensor_>>);

impl Tensor_ {
    pub fn default() -> Self {
        Self {
            id: TensorId::new(),
            name: String::new(),
            dtype: DataType::U8,
            layout: Layout::default(),
            storage: None,
            src_tensor: Vec::new(),
            length: 0,
            tensor_type: TensorType::UNKNOWN,
            view_offs: 0,
            op_type: TensorOpType::UNKNOWN,
        }
    }

    pub fn set_tensor_id(&mut self, id: TensorId) -> &mut Self {
        self.id = id;
        self
    }

    pub fn get_tensor_id(&self) -> &TensorId {
        &self.id
    }

    pub fn set_src_tensor(&mut self, src_tensor: Vec<Tensor>) -> &mut Self {
        self.src_tensor = src_tensor;
        self
    }

    pub fn get_src_tensor(&self) -> &Vec<Tensor> {
        &self.src_tensor
    }

    pub fn set_data_type(&mut self, dtype: DataType) -> &mut Self {
        self.dtype = dtype;
        self
    }

    pub fn set_shape(&mut self, shape: Shape) -> &mut Self {
        self.layout.shape = shape;
        self
    }

    pub fn get_shape(&self) -> &Shape {
        &self.layout.shape
    }

    pub fn set_length(&mut self, length: usize) -> &mut Self {
        self.length = length;
        self
    }

    pub fn get_length(&self) -> usize {
        self.length
    }

    pub fn get_dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn set_name(&mut self, name: String) -> &mut Self {
        self.name = name;
        self
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn set_data(&mut self, data: Arc<MemoryBlock>) -> &mut Self {
        self.storage = Some(data);
        self
    }

    pub fn set_op_type(&mut self, op_type: TensorOpType) -> &mut Self {
        self.op_type = op_type;
        self
    }

    pub fn get_op_type(&self) -> &TensorOpType {
        &self.op_type
    }

    pub fn get_data(&self) -> Result<*mut u8, String> {
        // self.storage
        //     .as_ref()
        //     .ok_or("Tensor storage is None".to_string())
        todo!("get_data");
    }
}

impl Tensor {
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let mut tensor = Tensor_::default();
        tensor.set_op_type(TensorOpType::TensorOpMul);
        tensor.set_src_tensor(vec![self.clone(), other.clone()]);
        Tensor(Arc::new(RefCell::new(tensor)))
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
        let tensor = Tensor_::default();

        assert_eq!(tensor.get_name(), &String::new());
        assert_eq!(tensor.get_dtype(), &DataType::U8);
        assert_eq!(tensor.get_length(), 0);
        assert_eq!(tensor.tensor_type, TensorType::UNKNOWN);
        assert_eq!(tensor.get_op_type(), &TensorOpType::UNKNOWN);
        assert_eq!(tensor.view_offs, 0);
    }

    #[test]
    fn test_tensor_setters() {
        let mut tensor = Tensor_::default();

        let name = "test_tensor".to_string();
        tensor.set_name(name.clone());
        assert_eq!(tensor.get_name(), &name);

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
            assert_eq!(tensor.get_dtype(), &dtype);
        }

        let shape = Shape([2, 3, 4, 5]);
        tensor.set_shape(shape);
        assert_eq!(tensor.get_shape(), &shape);

        tensor.set_length(100);
        assert_eq!(tensor.get_length(), 100);

        for op_type in
            [TensorOpType::UNKNOWN, TensorOpType::TensorOpView, TensorOpType::TensorOpMul]
        {
            tensor.set_op_type(op_type);
            assert_eq!(tensor.get_op_type(), &op_type);
        }

        for tensor_type in [TensorType::UNKNOWN, TensorType::InputParam, TensorType::OutputParam] {
            tensor.tensor_type = tensor_type;
            assert_eq!(tensor.tensor_type, tensor_type);
        }
    }

    #[test]
    fn test_setter_chaining() {
        let mut tensor = Tensor_::default();

        let _ = tensor
            .set_name("chained".to_string())
            .set_data_type(DataType::F32)
            .set_shape(Shape([1, 2, 3, 4]))
            .set_length(42);

        assert_eq!(tensor.get_name(), &"chained".to_string());
        assert_eq!(tensor.get_dtype(), &DataType::F32);
        assert_eq!(tensor.get_shape(), &Shape([1, 2, 3, 4]));
        assert_eq!(tensor.get_length(), 42);
    }

    #[test]
    fn test_tensor_src_tensor() {
        let mut tensor = Tensor_::default();

        let t1 = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let t2 = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let t3 = Tensor(Arc::new(RefCell::new(Tensor_::default())));

        tensor.set_src_tensor(vec![t1.clone(), t2.clone()]);

        let src = tensor.get_src_tensor();
        assert_eq!(src.len(), 2);

        tensor.set_src_tensor(vec![t3]);
        assert_eq!(tensor.get_src_tensor().len(), 1);
    }

    #[test]
    fn test_tensor_data() {
        let tensor = Tensor_::default();
        assert!(tensor.storage.is_none());
    }

    #[test]
    fn test_tensor_creation_and_deref() {
        let inner = Tensor_::default();
        let tensor = Tensor(Arc::new(RefCell::new(inner)));

        let ref_cell: &RefCell<Tensor_> = &*tensor;
        let borrowed = ref_cell.borrow();

        assert_eq!(borrowed.get_name(), &String::new());
        assert_eq!(borrowed.get_length(), 0);
    }

    #[test]
    fn test_tensor_clone() {
        let t1 = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        t1.borrow_mut().set_name("original".to_string());

        let t2 = t1.clone();

        assert_eq!(t1.borrow().get_name(), t2.borrow().get_name());

        t2.borrow_mut().set_name("copy".to_string());
        assert_eq!(t1.borrow().get_name(), &"copy".to_string());
        assert_eq!(t2.borrow().get_name(), &"copy".to_string());
    }

    #[test]
    fn test_tensor_mul() {
        let t1 = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let t2 = Tensor(Arc::new(RefCell::new(Tensor_::default())));

        t1.borrow_mut().set_name("tensor_a".to_string());
        t2.borrow_mut().set_name("tensor_b".to_string());

        let result = t1.mul(&t2);

        let inner = result.borrow();

        assert_eq!(inner.get_op_type(), &TensorOpType::TensorOpMul);

        let src = inner.get_src_tensor();
        assert_eq!(src.len(), 2);

        assert_eq!(src[0].borrow().get_name(), &"tensor_a".to_string());
        assert_eq!(src[1].borrow().get_name(), &"tensor_b".to_string());
    }

    #[test]
    fn test_tensor_mul_graph() {
        let a = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let b = Tensor(Arc::new(RefCell::new(Tensor_::default())));
        let c = Tensor(Arc::new(RefCell::new(Tensor_::default())));

        let ab = a.mul(&b);
        let abc = ab.mul(&c);

        let abc_inner = abc.borrow();
        assert_eq!(abc_inner.get_op_type(), &TensorOpType::TensorOpMul);
        assert_eq!(abc_inner.get_src_tensor().len(), 2);

        let ab_inner = abc_inner.get_src_tensor()[0].borrow();
        assert_eq!(ab_inner.get_op_type(), &TensorOpType::TensorOpMul);
        assert_eq!(ab_inner.get_src_tensor().len(), 2);
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
            for j in (i + 1)..types.len() {
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
        let mut tensor = Tensor_::default();
        let new_id = TensorId::new();

        tensor.set_tensor_id(new_id);
        assert_eq!(tensor.get_tensor_id(), &new_id);
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
            let mut tensor = Tensor_::default();
            tensor.set_data_type(dtype);
            assert_eq!(tensor.get_dtype(), &dtype);
        }
    }
}
