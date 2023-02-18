use crate::{Distance, DistanceCmp, Embedding, EmbeddingProvider};

pub struct VecDotDistance {}

pub const VEC_DOT_DISTANCE: VecDotDistance = VecDotDistance {};

impl Distance<&Vec<f64>> for VecDotDistance {
    fn distance_cmp(&self, a: Embedding<&Vec<f64>>, b: Embedding<&Vec<f64>>) -> DistanceCmp {
        let res: f64 = a
            .value
            .iter()
            .zip(b.value.iter())
            .map(|(&cur_a, &cur_b)| cur_a * cur_b)
            .sum();
        DistanceCmp::of((-res).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
    }
}

pub struct VecL2Distance {}

pub const VEC_L2_DISTANCE: VecL2Distance = VecL2Distance {};

impl Distance<&Vec<f64>> for VecL2Distance {
    fn distance_cmp(&self, a: Embedding<&Vec<f64>>, b: Embedding<&Vec<f64>>) -> DistanceCmp {
        let res: f64 = a
            .value
            .iter()
            .zip(b.value.iter())
            .map(|(&cur_a, &cur_b)| (cur_a - cur_b) * (cur_a - cur_b))
            .sum();
        DistanceCmp::of(res)
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to().sqrt()
    }
}

pub struct VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    embeddings: Vec<Vec<f64>>,
    distance: &'a D,
}

impl<'a, D> VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    pub fn new(embeddings: Vec<Vec<f64>>, distance: &'a D) -> Self {
        VecProvider {
            embeddings,
            distance,
        }
    }
}

impl<'a, D> EmbeddingProvider<'a, D, &'a Vec<f64>> for VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    fn get_embed(&'a self, index: usize) -> &'a Vec<f64> {
        &self.embeddings[index]
    }

    fn all(&self) -> std::ops::Range<usize> {
        0..self.embeddings.len()
    }

    fn distance(&self) -> &'a D {
        self.distance
    }
}

// impl<'a, D> NearestNeighbors<&Vec<f64>> for VecProvider<'a, D>
// where
//     D: Distance<&'a Vec<f64>>,
// {
//     fn get_closest(&mut self, embed: Embedding<&Vec<f64>>, count: usize) -> Vec<(usize, f64)> {
//         let distance = self.distance;
//         let dists: Vec<DistanceCmp> = self
//             .all()
//             .map(|ix| distance.distance_cmp(self.get(ix), embed))
//             .collect();
//         let mut indices: Vec<usize> = (0..dists.len()).collect();
//         indices.sort_unstable();
//         indices
//             .iter()
//             .take(count)
//             .map(|&ix| (ix, distance.finalize_distance(&dists[ix])))
//             .collect()
//     }
// }
