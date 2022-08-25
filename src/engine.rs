use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

///
/// Scalar tensor is a single value object that stores it's
/// data and it's gradient.
/// In order for backpropogation to work correctly we also need to store
/// the building blocks of this tensor - the so called children.
#[derive(Debug)]
pub struct ScalarTensor {
    pub data: f32,
    pub grad: f32,
    children: Vec<MutableScalarTensor>,
    op: Op,
    unique_id: usize,
}

type MutableScalarTensor = Rc<RefCell<ScalarTensor>>;

#[derive(Debug, PartialEq)]
enum Op {
    NONE,
    ADD,
    MUL,
}

impl Display for ScalarTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let op_string = match self.op {
            Op::NONE => "",
            Op::ADD => "+",
            Op::MUL => "*",
        };
        write!(
            f,
            "(data: {} | grad: {} | op: {})",
            self.data, self.grad, op_string
        )
    }
}

impl Hash for ScalarTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique_id.hash(state);
    }
}

impl PartialEq for ScalarTensor {
    fn eq(&self, other: &Self) -> bool {
        self.unique_id == other.unique_id
    }
}

impl Eq for ScalarTensor {}

fn add_to_children(tensor: &MutableScalarTensor, tensors_to_add: &Vec<MutableScalarTensor>) {
    let mut tmp = tensor.borrow_mut();
    for c in tensors_to_add.iter() {
        tmp.children.push(c.clone());
    }
}

pub fn add(rhs: &MutableScalarTensor, lhs: &MutableScalarTensor) -> MutableScalarTensor {
    let new_tensor = ScalarTensor::new_with_op(rhs.borrow().data + lhs.borrow().data, Op::ADD);
    add_to_children(&new_tensor, &vec![rhs.clone(), lhs.clone()]);
    new_tensor
}

pub fn mul(rhs: &MutableScalarTensor, lhs: &MutableScalarTensor) -> MutableScalarTensor {
    let new_tensor = ScalarTensor::new_with_op(rhs.borrow().data * lhs.borrow().data, Op::MUL);
    add_to_children(&new_tensor, &vec![rhs.clone(), lhs.clone()]);
    new_tensor
}

fn get_topo_graph_for_tensor(
    tensor: &MutableScalarTensor,
    visited: &mut HashSet<usize>,
    topo_graph: &mut Vec<MutableScalarTensor>,
) {
    if visited.contains(&tensor.borrow().unique_id) {
        return;
    }

    visited.insert(tensor.borrow().unique_id);
    for child in tensor.borrow().children.iter() {
        get_topo_graph_for_tensor(child, visited, topo_graph);
    }
    topo_graph.push(tensor.clone());
}

fn get_reversed_topo_graph(tensor: &MutableScalarTensor) -> Vec<MutableScalarTensor> {
    let mut topo_graph = Vec::new();
    get_topo_graph_for_tensor(tensor, &mut HashSet::new(), &mut topo_graph);
    topo_graph.reverse();
    topo_graph
}

// Apply back backpropagation to this tensor
pub fn backward(tensor: &MutableScalarTensor) {
    // First we initialize the root node with gradient of 1.0
    tensor.borrow_mut().grad = 1.0;
    let topo_graph = get_reversed_topo_graph(tensor);
    for t in topo_graph.iter() {
        // Now apply the grad function on each tensor
        t.borrow().grad_fn();
    }
}

impl ScalarTensor {
    fn new_with_op(data: f32, op: Op) -> MutableScalarTensor {
        Rc::new(RefCell::new(Self {
            data,
            grad: 0.0,
            children: Vec::new(),
            op: op,
            unique_id: OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst),
        }))
    }

    pub fn new(data: f32) -> MutableScalarTensor {
        Self::new_with_op(data, Op::NONE)
    }

    fn grad_fn(&self) {
        match self.op {
            Op::NONE => (),
            Op::ADD => {
                for child in self.children.iter() {
                    // Local derivative of an addition is 1, and we apply the chain rule by
                    // multiplying by the total grad so far
                    child.borrow_mut().grad += self.grad * 1.0;
                }
            }
            Op::MUL => {
                // TODO: make this safe (should do for everything else as well)
                // Local derivative of multiplication is the number we are
                // multiplying by so
                let mut left = self.children[0].borrow_mut();
                let mut right = self.children[1].borrow_mut();
                left.grad += right.data * self.grad;
                right.grad += left.data * self.grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let a = ScalarTensor::new(5.0);
        let t = a.borrow();
        assert_eq!(5.0, t.data);
        assert_eq!(0, t.children.len());
        assert_eq!(Op::NONE, t.op);
    }

    #[test]
    fn test_addition() {
        let t1 = ScalarTensor::new(5.0);
        let t2 = ScalarTensor::new(3.14);
        let out_tensor = add(&t1, &t2);
        let out = out_tensor.borrow();
        assert_eq!(8.14, out.data);
        assert_eq!(2, out.children.len());
        assert_eq!(Op::ADD, out.op);
    }

    #[test]
    fn test_grad_fn_only_add() {
        let t1 = ScalarTensor::new(5.0);
        let t2 = ScalarTensor::new(3.14);
        let out = add(&t1, &t2);
        assert_eq!(0.0, t1.borrow().grad);
        assert_eq!(0.0, t2.borrow().grad);
        out.borrow_mut().grad = 1.0;
        out.borrow().grad_fn();
        assert_eq!(1.0, t1.borrow().grad);
        assert_eq!(1.0, t2.borrow().grad);
    }

    #[test]
    fn test_topo_graph() {
        let t1 = ScalarTensor::new(123.0);
        let t2 = ScalarTensor::new(11.0);
        let out1 = add(&t1, &t2);
        let out2 = add(&t1, &t2);
        let out3 = add(&out1, &out2);

        let topo_graph = get_reversed_topo_graph(&out3);
        let expected_vec = vec![
            out3.borrow(),
            out2.borrow(),
            out1.borrow(),
            t2.borrow(),
            t1.borrow(),
        ];

        assert_eq!(expected_vec.len(), topo_graph.len());
        for (expected, real) in expected_vec.iter().zip(topo_graph.iter()) {
            assert_eq!(expected.unique_id, real.borrow().unique_id);
        }
    }

    #[test]
    fn test_backward_add_only() {
        let t1 = ScalarTensor::new(123.0);
        let t2 = ScalarTensor::new(11.0);
        let out1 = add(&t1, &t2);
        let out2 = add(&t1, &t2);
        let out3 = add(&out1, &out2);

        backward(&out3);
        for t in vec![out1, out2, out3] {
            assert_eq!(1.0, t.borrow().grad);
        }

        // t1 and t2 are both contributing to 2 other tensors, so their gradient should
        // be 2.0. This can be shown as
        // out3 = out2 + out1 = (t1 + t2) + (t1 + t2)
        // Thus if we increase t1 by h for example we get
        // (t1 + h + t2) + (t1 +h + t2) = out3 + 2h
        // a change in t1 will have double the change on out3
        for t in vec![t1, t2] {
            assert_eq!(2.0, t.borrow().grad);
        }
    }

    #[test]
    fn test_mul() {
        let t1 = ScalarTensor::new(5.0);
        let t2 = ScalarTensor::new(6.5);
        let out_tensor = mul(&t1, &t2);
        let out = out_tensor.borrow();
        assert_eq!(32.5, out.data);
        assert_eq!(2, out.children.len());
        assert_eq!(Op::MUL, out.op);
    }

    #[test]
    fn test_grad_fn_only_mul() {
        let t1 = ScalarTensor::new(3.0);
        let t2 = ScalarTensor::new(7.2);
        let out = mul(&t1, &t2);
        assert_eq!(0.0, t1.borrow().grad);
        assert_eq!(0.0, t2.borrow().grad);
        out.borrow_mut().grad = 1.0;
        out.borrow().grad_fn();
        assert_eq!(7.2, t1.borrow().grad);
        assert_eq!(3.0, t2.borrow().grad);
    }

    #[test]
    fn test_backward_mul_only_simple() {
        let t1 = ScalarTensor::new(2.0);
        let t2 = ScalarTensor::new(3.0);
        let out1 = mul(&t1, &t2);
        let out2 = mul(&t1, &t2);
        let out3 = mul(&out1, &out2);

        backward(&out3);

        // The equation is (t1*t2)*(t1*t2)
        assert_eq!(out3.borrow().grad, 1.0);
        assert_eq!(out2.borrow().grad, 2.0 * 3.0);
        assert_eq!(out1.borrow().grad, 2.0 * 3.0);

        // f(x+h) - f(x) / h = the derivative, so dt1((t1*t2)*(t1*t2))/dout3 = 2t1*t2**2
        assert_eq!(t1.borrow().grad, 2.0 * 2.0 * 3.0 * 3.0);
        assert_eq!(t2.borrow().grad, 2.0 * 3.0 * 2.0 * 2.0);
    }

    #[test]
    fn test_backward_mul_only() {
        let t1 = ScalarTensor::new(123.321);
        let t2 = ScalarTensor::new(321.123);
        let out1 = mul(&t1, &t2);
        let out2 = mul(&t1, &t2);
        let out3 = mul(&out1, &out2);

        backward(&out3);

        // The equation is (t1*t2)*(t1*t2)
        assert_eq!(out3.borrow().grad, 1.0);
        assert_eq!(out2.borrow().grad, 123.321 * 321.123);
        assert_eq!(out1.borrow().grad, 123.321 * 321.123);
        assert_eq!(t1.borrow().grad, 2.0 * 321.123 * (123.321 * 321.123));
        assert_eq!(t2.borrow().grad, 2.0 * 123.321 * (123.321 * 321.123));
    }
}
