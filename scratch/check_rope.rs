use burn::nn::{RotaryEncoding, RotaryEncodingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

fn check_api<B: Backend>(device: &B::Device) {
    let config = RotaryEncodingConfig::new(64);
    let rope = config.init::<B>(device);
    
    let x = Tensor::<B, 4>::zeros([1, 1, 10, 64], device);
    let y = rope.forward(x);
}

fn main() {}
