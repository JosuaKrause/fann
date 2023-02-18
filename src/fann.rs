use rayon::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    iter::repeat,
    marker::PhantomData,
};

use crate::{Cache, Distance, DistanceCmp, Embedding, EmbeddingProvider, NearestNeighbors};

const HIGHLIGHT_A: &str = "*";
const HIGHLIGHT_B: &str = ":";
const NO_HIGHLIGHT: &str = "";

struct Child<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    node: Node<'a, E, D, C, T>,
    center_dist: DistanceCmp,
}

struct Node<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    centroid_index: usize,
    radius: DistanceCmp,
    count: usize,
    children: Vec<Child<'a, E, D, C, T>>,
    provider_type: PhantomData<&'a E>,
    distance_type: PhantomData<D>,
    cache_type: PhantomData<C>,
    embed_type: PhantomData<T>,
}

impl<'a, E, D, C, T> Node<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    fn new(centroid_index: usize) -> Self {
        Node {
            centroid_index,
            radius: DistanceCmp::zero(),
            count: 1,
            children: Vec::new(),
            provider_type: PhantomData,
            distance_type: PhantomData,
            cache_type: PhantomData,
            embed_type: PhantomData,
        }
    }

    fn get_embed(&self, provider: &'a E) -> Embedding<T> {
        provider.get(self.centroid_index)
    }

    fn count_descendants(&self) -> usize {
        self.count
    }

    fn count_children(&self) -> usize {
        self.children.len()
    }

    fn get_dist(&self, embed: &Embedding<T>, provider: &'a E, cache: &mut C) -> DistanceCmp {
        let distance = provider.distance();
        cache.cached_distance(&self.get_embed(provider), embed, distance)
    }

    fn get_dist_min(&self, embed: &Embedding<T>, provider: &'a E, cache: &mut C) -> DistanceCmp {
        let dist = self.get_dist(embed, provider, cache);
        dist.combine(&self.radius, |d, radius| f64::max(0.0, d - radius))
    }

    fn get_child_dist_max(child: &Child<'a, E, D, C, T>) -> DistanceCmp {
        child
            .center_dist
            .combine(&child.node.radius, |center_dist, radius| {
                center_dist + radius
            })
    }

    fn compute_radius(&mut self) {
        self.radius = self
            .children
            .iter()
            .map(|child| Node::get_child_dist_max(child))
            .max()
            .unwrap_or(DistanceCmp::zero());
    }

    fn add_child(&mut self, child: Node<'a, E, D, C, T>, provider: &'a E, cache: &mut C) {
        let center_dist = self.get_dist(&child.get_embed(provider), provider, cache);
        self.count += child.count_descendants();
        self.children.push(Child {
            node: child,
            center_dist,
        });
        self.children
            .sort_unstable_by(|a, b| a.center_dist.cmp(&b.center_dist).reverse());
    }

    fn get_closest(
        &self,
        res: &mut Vec<(usize, DistanceCmp)>,
        embed: &Embedding<T>,
        count: usize,
        provider: &'a E,
        cache: &mut C,
    ) {
        let own_dist = self.get_dist(embed, provider, cache);

        fn max_dist(res: &Vec<(usize, DistanceCmp)>, count: usize) -> DistanceCmp {
            let index = count.min(res.len()) - 1;
            res[index].1
        }

        fn add_node<'a, E, D, C, T>(
            res: &mut Vec<(usize, DistanceCmp)>,
            node: &Node<'a, E, D, C, T>,
            distance: DistanceCmp,
            count: usize,
        ) where
            E: EmbeddingProvider<'a, D, T>,
            D: Distance<T> + Copy,
            C: Cache,
            T: 'a,
        {
            let element = (node.centroid_index, distance);
            let mindex = res.binary_search_by(|&(_, dist)| dist.cmp(&distance));
            match mindex {
                Ok(index) => res.insert(index, element),
                Err(index) => res.insert(index, element),
            }
            res.truncate(count);
        }

        if res.len() < count || own_dist < max_dist(res, count) {
            add_node(res, self, own_dist, count);
        }
        if self.radius < own_dist {
            for child in self.children.iter() {
                if max_dist(res, count)
                    < own_dist.combine(&child.center_dist, |own, center| own - center)
                {
                    continue;
                }
                child.node.get_closest(res, embed, count, provider, cache);
            }
        } else {
            let mut inners: Vec<(&Node<E, D, C, T>, DistanceCmp)> = self
                .children
                .iter()
                .map(|child| (&child.node, child.node.get_dist_min(embed, provider, cache)))
                .collect::<Vec<(&Node<E, D, C, T>, DistanceCmp)>>();
            inners.sort_unstable_by(|(_, dist_a), (_, dist_b)| dist_a.cmp(dist_b));
            for (cnode, cmin) in inners.iter() {
                if max_dist(res, count) < *cmin {
                    continue;
                }
                cnode.get_closest(res, embed, count, provider, cache);
            }
        }
    }

    fn draw(
        &self,
        pad: usize,
        show_ixs: &HashMap<usize, bool>,
        stats: &HashMap<usize, &str>,
        prune: bool,
        radius: bool,
    ) -> Vec<String> {
        let highlight = match show_ixs.get(&self.centroid_index) {
            Some(true) => HIGHLIGHT_A,
            Some(false) => HIGHLIGHT_B,
            None => NO_HIGHLIGHT,
        };
        let num = format!(
            "{ix: >pad$}",
            ix = self.centroid_index,
            pad = pad - highlight.len(),
        );
        let rad = if radius {
            format!("[r:{r}]", r = self.radius.to())
        } else {
            "".to_owned()
        };
        let own = format!(
            "({highlight}{num}{rad})",
            highlight = highlight,
            num = num,
            rad = rad,
        );
        let children_count = self.count_children();
        if children_count == 0 {
            return Vec::from([own]);
        }
        if !radius && self.count_descendants() == children_count + 1 {
            let mut chs = self
                .children
                .iter()
                .map(|child| {
                    let cix = child.node.centroid_index;
                    let chighlight = match show_ixs.get(&cix) {
                        Some(true) => HIGHLIGHT_A,
                        Some(false) => HIGHLIGHT_B,
                        None => NO_HIGHLIGHT,
                    };
                    format!("{chighlight}{cix}", chighlight = chighlight, cix = cix)
                })
                .collect::<Vec<String>>()
                .join(", ");
            if prune && !chs.contains(&HIGHLIGHT_A) && !chs.contains(&HIGHLIGHT_B) {
                chs = "...".to_string();
            }
            return Vec::from([format!("{own}━({chs})", own = own, chs = chs)]);
        }
        let bar = repeat(" ").take(own.len()).collect::<String>();
        let sown = own.as_str();
        let sbar = bar.as_str();
        self.children
            .iter()
            .map(|child| {
                (
                    child.node.centroid_index,
                    child.node.draw(pad, &show_ixs, &stats, prune, radius),
                )
            })
            .enumerate()
            .map(|(cix, (child_ix, mut lines))| {
                let all_lines = lines.join("");
                if prune && !all_lines.contains(&HIGHLIGHT_A) && !all_lines.contains(&HIGHLIGHT_B) {
                    lines = Vec::from(["(...)".to_owned()]);
                }
                let cix = cix;
                let child_ix = child_ix;
                lines.into_iter().enumerate().map(move |(lix, line)| {
                    let start = if lix == 0 && cix == 0 { sown } else { sbar };
                    let mid: String = if lix == 0 {
                        let mid = if cix == 0 {
                            if children_count > 1 {
                                "┳"
                            } else {
                                "━"
                            }
                        } else {
                            if cix >= children_count - 1 {
                                "┗"
                            } else {
                                "┣"
                            }
                        };
                        match stats.get(&child_ix) {
                            Some(state) => state.to_uppercase().chars().nth(0).unwrap().to_string(),
                            None => mid.to_owned(),
                        }
                    } else {
                        if cix >= children_count - 1 {
                            " "
                        } else {
                            "┃"
                        }
                        .to_owned()
                    };
                    format!("{start}{mid}{line}", start = start, mid = mid, line = line)
                })
            })
            .flatten()
            .collect::<Vec<String>>()
    }
}

pub struct Fann<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    provider: &'a E,
    root: Option<Node<'a, E, D, C, T>>,
    high_ix: usize,
}

impl<'a, E, D, C, T> Fann<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    pub fn new(provider: &'a E) -> Self {
        Fann {
            provider,
            root: None,
            high_ix: 0,
        }
    }

    fn get_dist(
        &self,
        embed_a: &Embedding<T>,
        embed_b: &Embedding<T>,
        cache: &mut C,
    ) -> DistanceCmp {
        cache.cached_distance(embed_a, embed_b, self.provider.distance())
    }

    fn centroid(&self, all_ixs: &Vec<usize>, cache: &mut C) -> usize {
        let provider = self.provider;
        let (res_ix, _) =
            all_ixs
                .iter()
                .fold((None, DistanceCmp::of(f64::INFINITY)), |best, &ix| {
                    let (best_ix, best_dist) = best;
                    let embed = provider.get(ix);
                    let cur_dist: DistanceCmp =
                        all_ixs.iter().fold(DistanceCmp::zero(), |res, &oix| {
                            if oix == ix || res > best_dist {
                                res
                            } else {
                                let oembed = provider.get(oix);
                                res.combine(&self.get_dist(&embed, &oembed, cache), |cur, dist| {
                                    cur + dist
                                })
                            }
                        });
                    if best_ix.is_none() || cur_dist < best_dist {
                        (Some(ix), cur_dist)
                    } else {
                        best
                    }
                });
        res_ix.unwrap()
    }

    fn kmedoid(
        &self,
        all_ixs: Vec<usize>,
        k_num: usize,
        cache: &mut C,
    ) -> Vec<(usize, Vec<usize>)> {
        if all_ixs.len() <= k_num {
            return all_ixs.iter().map(|&ix| (ix, Vec::new())).collect();
        }
        let provider = self.provider;
        let buff_size = 10;
        let mut rounds = 1000;
        let mut buff: VecDeque<Vec<usize>> = VecDeque::with_capacity(buff_size);
        buff.push_front(all_ixs[..k_num].to_vec());
        let mut done = false;
        loop {
            let centroids: Vec<usize> = buff.get(0).unwrap().clone();
            let mut res: Vec<(usize, Vec<usize>)> =
                centroids.iter().map(|&ix| (ix, Vec::from([ix]))).collect();
            all_ixs
                .iter()
                .filter(|&ix| !centroids.contains(ix))
                .for_each(|&ix| {
                    let embed = provider.get(ix);
                    let (_, best) = res
                        .iter_mut()
                        .min_by(|(a, _), (b, _)| {
                            let dist_a = self.get_dist(&embed, &provider.get(*a), cache);
                            let dist_b = self.get_dist(&embed, &provider.get(*b), cache);
                            dist_a.cmp(&dist_b)
                        })
                        .unwrap();
                    best.push(ix);
                });
            if done {
                return res;
            }
            rounds -= 1;
            if rounds <= 0 {
                eprintln!("exhausted iteration steps");
                return res;
            }
            let new_cs: Vec<usize> = res
                .iter()
                .map(|(_, assignments)| self.centroid(assignments, cache))
                .collect();
            if buff.par_iter().any(|old_cs| *old_cs == new_cs) {
                // TODO use par for actually useful things
                done = true;
            }
            while buff.len() >= buff_size {
                buff.pop_back();
            }
            buff.push_front(new_cs.clone());
        }
    }

    pub fn build(&mut self, max_node_size: Option<usize>, cache: &mut C) {
        let mut all_ixs: Vec<usize> = self.provider.all().collect();
        let max_node_size = match max_node_size {
            Some(max_node_size) => max_node_size,
            None => all_ixs.len(),
        };
        self.high_ix = *all_ixs.iter().max().unwrap();
        let root_ix = self.centroid(&all_ixs, cache);

        fn remove(ixs: &mut Vec<usize>, index: usize) {
            ixs.retain(|&ix| ix != index);
        }

        fn build_level<'a, E, D, C, T>(
            tree: &Fann<'a, E, D, C, T>,
            cache: &mut C,
            cur_root_ix: usize,
            cur_all_ixs: Vec<usize>,
            max_node_size: usize,
        ) -> Node<'a, E, D, C, T>
        where
            E: EmbeddingProvider<'a, D, T>,
            D: Distance<T> + Copy,
            C: Cache,
            T: 'a,
        {
            let mut node: Node<'a, E, D, C, T> = Node::new(cur_root_ix);
            let num_k = if max_node_size * max_node_size > cur_all_ixs.len() {
                ((cur_all_ixs.len() as f64).sqrt() as usize).max(1)
            } else {
                max_node_size
            };
            if num_k == 1 || cur_all_ixs.len() <= num_k {
                cur_all_ixs.iter().for_each(|&ix| {
                    let mut cnode: Node<'a, E, D, C, T> = Node::new(ix);
                    cnode.compute_radius();
                    node.add_child(cnode, tree.provider, cache);
                });
            } else {
                tree.kmedoid(cur_all_ixs, num_k, cache)
                    .into_iter()
                    .for_each(|(centroid_ix, mut assignments)| {
                        remove(&mut assignments, centroid_ix);
                        let child_node =
                            build_level(tree, cache, centroid_ix, assignments, max_node_size);
                        node.add_child(child_node, tree.provider, cache);
                    });
            }
            node.compute_radius();
            node
        }

        remove(&mut all_ixs, root_ix);
        self.root = Some(build_level(self, cache, root_ix, all_ixs, max_node_size));
    }

    pub fn draw(&self, prune: bool, radius: bool) -> String {
        let pad = format!("{high_ix}", high_ix = self.high_ix).len();
        let show_ixs: HashMap<usize, bool> = HashMap::new();
        let stats: HashMap<usize, &str> = HashMap::new();
        self.root
            .as_ref()
            .unwrap()
            .draw(pad, &show_ixs, &stats, prune, radius)
            .join("\n")
    }
}

impl<'a, E, D, C, T> NearestNeighbors<C, T> for Fann<'a, E, D, C, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    C: Cache,
    T: 'a,
{
    fn get_closest(&self, embed: &Embedding<T>, count: usize, cache: &mut C) -> Vec<(usize, f64)> {
        let mut res: Vec<(usize, DistanceCmp)> = Vec::with_capacity(count + 1);
        self.root
            .as_ref()
            .unwrap()
            .get_closest(&mut res, embed, count, self.provider, cache);
        let distance = self.provider.distance();
        res.iter()
            .map(|(ix, v)| (*ix, distance.finalize_distance(v)))
            .collect()
    }
}
