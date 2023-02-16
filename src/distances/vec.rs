use crate::{Distance, Embedding};

pub struct DotDistance {
    embeddings: Vec<Vec<f64>>,
}

impl DotDistance {
    pub fn new(embeddings: Vec<Vec<f64>>) -> Self {
        DotDistance { embeddings }
    }
}

impl<'a> Distance<'a, &'a Vec<f64>> for DotDistance {
    fn distance(&self, a: &Embedding<&Vec<f64>>, b: &Embedding<&Vec<f64>>) -> f64 {
        let res: f64 = a
            .value
            .iter()
            .zip(b.value.iter())
            .map(|(&cur_a, &cur_b)| cur_a * cur_b)
            .sum();
        (-res).exp()
    }

    fn get(&'a self, index: usize) -> &'a Vec<f64> {
        self.embeddings.get(index).unwrap()
    }

    fn all(&self) -> std::ops::Range<usize> {
        0..self.embeddings.len()
    }
}
