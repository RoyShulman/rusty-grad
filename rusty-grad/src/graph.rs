use crate::engine::{MutableScalarTensor, ScalarTensor, ScalarTensorUniqueID};
use std::{collections::HashSet, fmt::Display};

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct ScalarTensorGraphNode {
    label: String,
    unique_id: ScalarTensorUniqueID,
}

impl ScalarTensorGraphNode {
    pub fn new(tensor: &ScalarTensor) -> Self {
        Self {
            label: tensor.to_string(),
            unique_id: tensor.unique_id,
        }
    }
}

#[derive(Debug, PartialEq)]
struct ScalarTensorGraphEdge {
    from: ScalarTensorGraphNode,
    to: ScalarTensorGraphNode,
}

impl ScalarTensorGraphEdge {
    pub fn new(from: &ScalarTensor, to: &ScalarTensor) -> Self {
        Self {
            from: ScalarTensorGraphNode::new(from),
            to: ScalarTensorGraphNode::new(to),
        }
    }
}

impl Display for ScalarTensorGraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

// Using petgraph was too hard... so we just write one
struct DotDrawableScalarTensorGraph {
    nodes: HashSet<ScalarTensorGraphNode>,
    edges: Vec<ScalarTensorGraphEdge>, // Should this also be a hashet?
}

impl DotDrawableScalarTensorGraph {
    pub fn new(edges: Vec<ScalarTensorGraphEdge>) -> Self {
        let nodes = Self::get_nodes(&edges);
        Self { nodes, edges }
    }

    fn get_nodes(edges: &Vec<ScalarTensorGraphEdge>) -> HashSet<ScalarTensorGraphNode> {
        let mut nodes_set = HashSet::new();
        for edge in edges.iter() {
            nodes_set.insert(edge.from.clone());
            nodes_set.insert(edge.to.clone());
        }
        nodes_set
    }

    // I guess we want to take ownership because after to dot it's useless?
    pub fn to_dot(self) -> String {
        let mut dot_graph = String::from("digraph {\n   rankdir=LR;\n");
        for node in self.nodes {
            dot_graph.push_str(&format!(
                "   {} [ label = \"{}\" shape=record ]\n",
                node.unique_id, node.label
            ));
        }

        for edge in self.edges {
            dot_graph.push_str(&format!(
                "   {} -> {} [ label = \"\" ]\n",
                edge.from.unique_id, edge.to.unique_id
            ));
        }
        dot_graph.push_str("}");
        dot_graph
    }
}

pub struct ScalarTensorGraph;

impl ScalarTensorGraph {
    pub fn draw_root(root: &MutableScalarTensor) -> String {
        let mut nodes = HashSet::new();
        let mut edges = Vec::new();
        ScalarTensorGraph::trace_tensor(root, &mut nodes, &mut edges);
        DotDrawableScalarTensorGraph::new(edges).to_dot()
    }

    // TOOD: make this function better, and not modify it's parameters
    fn trace_tensor(
        tensor: &MutableScalarTensor,
        seen_nodes: &mut HashSet<ScalarTensorUniqueID>,
        edges: &mut Vec<ScalarTensorGraphEdge>, // TODO: Should this be a hashset?
    ) {
        let current = tensor.borrow();
        if !seen_nodes.contains(&current.unique_id) {
            seen_nodes.insert(current.unique_id);

            for child in current.children.iter() {
                edges.push(ScalarTensorGraphEdge::new(&child.borrow(), &current));

                ScalarTensorGraph::trace_tensor(child, seen_nodes, edges);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_tensor() {
        let t1 = ScalarTensor::new(1.2);
        let t2 = ScalarTensor::new(3.4);
        let out = &t1 + &t2;

        let mut nodes = HashSet::new();
        let mut edges = Vec::new();

        ScalarTensorGraph::trace_tensor(&out, &mut nodes, &mut edges);
        {
            let t1 = t1.borrow();
            let t2 = t2.borrow();
            let out = out.borrow();
            let mut expected_nodes = HashSet::new();
            expected_nodes.insert(out.unique_id);
            expected_nodes.insert(t1.unique_id);
            expected_nodes.insert(t2.unique_id);
            assert_eq!(expected_nodes, nodes);

            let expected_edges = vec![
                ScalarTensorGraphEdge::new(&t1, &out),
                ScalarTensorGraphEdge::new(&t2, &out),
            ];

            for (expected, real) in expected_edges.iter().zip(edges.iter()) {
                assert_eq!(*expected, *real);
            }
        }
    }
}
