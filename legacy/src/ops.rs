#[derive(Clone)]
pub enum OpParams {
    None, // Mul, Add, Relu

    Gemm { trans_a: bool, trans_b: bool, alpha: f32, beta: f32 },

    Softmax { axis: i32 },

    Reshape { shape: [usize; 4] },
}
