// builtin

// external

// internal
use ndarray::{Array1, Array2, ArrayView2, Axis};
use super::{Activation, Weight};


pub fn compute_output_grad(output: &Vec<f32>, answer: &Vec<f32>, raw: &Array2<f32>, end_actf: &Activation) -> Array1<f32> {
    let mut loss_vec: Vec<f32> = Vec::new();
    for (i, out) in output.iter().enumerate() {
        let expected: &f32 = answer.get(i).expect("Failed to get vector value");
        let residual: f32 = expected - out;
        loss_vec.push(residual);
    }

    let loss: Array1<f32> = Array1::from_vec(loss_vec);
    let mut scale: Array1<f32> = raw.clone().remove_axis(Axis(0));
    // scale.mapv_inplace(end_actf.get_deriv());
    scale.mapv_inplace(|x| end_actf.deriv_of(x));
    scale *= -1.0;

    loss * scale
}

pub fn compute_hidden_grad(dim: (usize, usize), prev_grad: &Array1<f32>, input_act: &Array2<f32>) -> Array2<f32> {
    let grad_matrix: ArrayView2<f32> = prev_grad.view().insert_axis(Axis(0));
    let grads: ArrayView2<f32> = grad_matrix.broadcast(dim)
        .expect("Failed to broadcast gradient vector");

    let mut hidden_update: Array2<f32> = input_act.view().reversed_axes().broadcast(dim)
        .expect("Failed to broadcast input vector").to_owned();

    hidden_update = hidden_update * grads;
    
    hidden_update
}

pub fn compute_bias_grad(prev_grad: &Array1<f32>) -> Array1<f32> {
    prev_grad.clone()
}

pub fn compute_recurrence_grad(
    dim: (usize, usize), 
    prev_grad: &Array1<f32>, 
    prev_act: Option<&Array2<f32>>
) -> Array2<f32> {
    if prev_act.is_some() {
        let activation: &Array2<f32> = prev_act.expect("Failed to get previous activations");
        let grad_matrix: ArrayView2<f32> = prev_grad.view().insert_axis(Axis(0));
        let grads: ArrayView2<f32> = grad_matrix.broadcast(dim)
            .expect("Failed to broadcast gradient vector");

        let mut recurrence_update: Array2<f32> = activation.view().reversed_axes().broadcast(dim)
            .expect("Failed to broadcast previous inputs").to_owned();

        recurrence_update = recurrence_update * grads;

        return recurrence_update;
    } else {
        return Array2::zeros(dim);
    }
}

pub fn compute_backpropogated_grad(
    hidden_layer_weights: &Weight,
    hidden_actf: &Activation, 
    prev_grad: Array1<f32>, 
    raw_input: &Array2<f32>,
) -> Array1<f32> {
    let mut input_vec: Array1<f32> = raw_input.view().remove_axis(Axis(0)).to_owned();
    // input_vec.mapv_inplace(hidden_actf.get_deriv());
    input_vec.mapv_inplace(|x| hidden_actf.deriv_of(x));

    let weights: ArrayView2<f32> = hidden_layer_weights.get_weight_matrix();
    let grads: Array2<f32> = prev_grad.insert_axis(Axis(1));

    let result = weights.dot(&grads).remove_axis(Axis(1));

    result * input_vec
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn hidden_gradient_test(){
        let dim: (usize, usize) = (4, 2);

        let prev_grad_vec: Vec<f32> = vec![1.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let input_act_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_act: Array2<f32> = Array1::from_vec(input_act_vec).insert_axis(Axis(0));

        let gradient: Array2<f32> = compute_hidden_grad(dim, &prev_grad, &input_act);
    

        let ans_vec: Vec<f32> = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let ans: Array2<f32> = Array2::from_shape_vec(dim, ans_vec)
            .expect("Failed to create ans matrix");

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn bias_gradient_test(){
        let prev_grad_vec: Vec<f32> = vec![1.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let gradient: Array1<f32> = compute_bias_grad(&prev_grad);

        let ans_vec: Vec<f32> = vec![1.0, 2.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn recurrence_gradient_test() {
        let dim: (usize, usize) = (3, 3);

        let prev_grad_vec: Vec<f32> = vec![1.0, 1.0, 1.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let prev_act_vec: Vec<f32> = vec![1.0, 2.0, 3.0];
        let prev_act: Array2<f32> = Array1::from_vec(prev_act_vec).insert_axis(Axis(0));

        let act_option: Option<&Array2<f32>> = Some(&prev_act);

        let gradient: Array2<f32> = compute_recurrence_grad(dim, &prev_grad, act_option);

        let ans_vec: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let ans: Array2<f32> = Array2::from_shape_vec(dim, ans_vec)
            .expect("Failed to create ans matrix");

        println!("{:?}", gradient);
        assert_eq!(gradient, ans);
    }

    #[test]
    fn backprop_test() {

        let hidden_layer_weights: Weight = Weight::random((3, 2), 0.999, 1.0);
        let hidden_actf: Activation = Activation::relu();

        let prev_grad_vec: Vec<f32> = vec![2.0, 2.0];
        let prev_grad: Array1<f32> = Array1::from_vec(prev_grad_vec);

        let raw_input_vec: Vec<f32> = vec![1.0, -1.0, 1.0];
        let raw_input: Array2<f32> = Array1::from_vec(raw_input_vec).insert_axis(Axis(0));

        let mut grad: Array1<f32> = compute_backpropogated_grad(
            &hidden_layer_weights,
            &hidden_actf,
            prev_grad, 
            &raw_input
        );

        grad.mapv_inplace(|x| x.round());

        let ans_vec: Vec<f32> = vec![4.0, 0.0, 4.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", grad);

        assert_eq!(grad, ans);
    }

    #[test]
    fn output_grad_test() {

        let output: Vec<f32> = vec![0.0, 0.0];
        let answer: Vec<f32> = vec![4.0, 2.0];

        let raw_vec: Vec<f32> = vec![-1.0, 1.0];
        let raw_output: Array2<f32> = Array1::from_vec(raw_vec).insert_axis(Axis(0));

        let end_actf: Activation = Activation::relu();

        let mut grad: Array1<f32> = compute_output_grad(&output, &answer, &raw_output, &end_actf);

        grad.mapv_inplace(|x | x.round());

        let ans_vec: Vec<f32> = vec![0.0, -2.0];
        let ans: Array1<f32> = Array1::from_vec(ans_vec);

        println!("{:?}", grad);
        assert_eq!(grad, ans);
    }
}