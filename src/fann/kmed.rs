use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    iter::repeat,
};

use crate::{
    info::Info, Cache, Distance, DistanceCmp, Embedding, EmbeddingProvider, LocalCache, Tree,
};

const HIGHLIGHT_A: &str = "*";
const HIGHLIGHT_B: &str = ":";
const NO_HIGHLIGHT: &str = "";

#[derive(Serialize, Deserialize)]
struct Child {
    node: Node,
    center_dist: DistanceCmp,
}

#[derive(Serialize, Deserialize)]
struct Node {
    centroid_index: usize,
    radius: DistanceCmp,
    count: usize,
    children: Vec<Child>,
}

impl Node {
    fn new(centroid_index: usize) -> Self {
        Node {
            centroid_index,
            radius: DistanceCmp::zero(),
            count: 1,
            children: Vec::new(),
        }
    }

    fn get_embed<'a, E, D, T>(&self, provider: &'a E) -> Embedding<T>
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
    {
        provider.get(self.centroid_index)
    }

    fn count_descendants(&self) -> usize {
        self.count
    }

    fn count_children(&self) -> usize {
        self.children.len()
    }

    fn get_internal_dist<'a, E, D, T, C, I>(
        &self,
        embed: &Embedding<T>,
        provider: &'a E,
        cache: &mut C,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
        let distance = provider.distance();
        cache.cached_distance(&self.get_embed(provider), embed, distance, info)
    }

    fn get_dist<'a, E, D, T, I, L>(
        &self,
        provider: &'a E,
        cache: &mut L,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        I: Info,
        L: LocalCache<'a, D, T>,
    {
        let distance = provider.distance();
        cache.cached_distance(&self.get_embed(provider), distance, info)
    }

    fn get_dist_min<'a, E, D, T, I, L>(
        &self,
        provider: &'a E,
        cache: &mut L,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        I: Info,
        L: LocalCache<'a, D, T>,
    {
        let dist = self.get_dist(provider, cache, info);
        dist.combine(&self.radius, |d, radius| f64::max(0.0, d - radius))
    }

    fn get_child_dist_max(child: &Child) -> DistanceCmp {
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

    fn add_child<'a, E, D, T, C, I>(
        &mut self,
        child: Node,
        provider: &'a E,
        cache: &mut C,
        info: &mut I,
    ) where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
        let center_dist = self.get_internal_dist(&child.get_embed(provider), provider, cache, info);
        self.count += child.count_descendants();
        self.children.push(Child {
            node: child,
            center_dist,
        });
        self.children
            .sort_unstable_by(|a, b| a.center_dist.cmp(&b.center_dist).reverse());
    }

    fn get_closest<'a, E, D, T, I, L>(
        &self,
        res: &mut Vec<(usize, DistanceCmp)>,
        count: usize,
        provider: &'a E,
        cache: &mut L,
        info: &mut I,
    ) where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        I: Info,
        L: LocalCache<'a, D, T>,
    {
        let own_dist = self.get_dist(provider, cache, info);

        fn max_dist(res: &Vec<(usize, DistanceCmp)>, count: usize) -> DistanceCmp {
            let index = count.min(res.len()) - 1;
            res[index].1
        }

        fn add_node(
            res: &mut Vec<(usize, DistanceCmp)>,
            node: &Node,
            distance: DistanceCmp,
            count: usize,
        ) {
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
        let is_outer = self.radius < own_dist;
        info.log_scan(self.centroid_index, is_outer);
        if is_outer {
            for child in self.children.iter() {
                let c_dist_est = own_dist.combine(&child.center_dist, |own, center| own - center);
                if max_dist(res, count) < c_dist_est {
                    continue;
                }
                child.node.get_closest(res, count, provider, cache, info);
            }
        } else {
            let mut inners: Vec<(&Node, DistanceCmp)> = self
                .children
                .iter()
                .map(|child| (&child.node, child.node.get_dist_min(provider, cache, info)))
                .collect();
            inners.sort_unstable_by(|(_, dist_a), (_, dist_b)| dist_a.cmp(dist_b));
            for (cnode, cmin) in inners.iter() {
                if max_dist(res, count) < *cmin {
                    continue;
                }
                cnode.get_closest(res, count, provider, cache, info);
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
        let bar: String = repeat(" ").take(own.len()).collect();
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

#[derive(Serialize, Deserialize)]
pub struct FannTree {
    root: Node,
    hash: String,
    distance_name: String,
}

impl FannTree {
    fn get_dist<'a, E, D, T, C, I>(
        provider: &E,
        embed_a: &Embedding<T>,
        embed_b: &Embedding<T>,
        cache: &mut C,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
        cache.cached_distance(embed_a, embed_b, provider.distance(), info)
    }

    fn centroid<'a, E, D, T, C, I>(
        provider: &'a E,
        all_ixs: &Vec<usize>,
        cache: &mut C,
        info: &mut I,
    ) -> usize
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
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
                                res.combine(
                                    &Self::get_dist(provider, &embed, &oembed, cache, info),
                                    |cur, dist| cur + dist,
                                )
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

    fn kmedoid<'a, E, D, T, C, I>(
        provider: &'a E,
        all_ixs: Vec<usize>,
        init_centroids: Option<Vec<usize>>,
        k_num: usize,
        cache: &mut C,
        info: &mut I,
    ) -> Vec<(usize, Vec<usize>)>
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
        if all_ixs.len() <= k_num {
            return all_ixs.iter().map(|&ix| (ix, Vec::new())).collect();
        }
        let buff_size = 10;
        let mut rounds = 1000;
        let mut buff: VecDeque<Vec<usize>> = VecDeque::with_capacity(buff_size);
        if let Some(init_centroids) = init_centroids {
            buff.push_front(init_centroids);
        } else {
            buff.push_front(all_ixs[..k_num].to_vec());
        }
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
                            let dist_a =
                                Self::get_dist(provider, &embed, &provider.get(*a), cache, info);
                            let dist_b =
                                Self::get_dist(provider, &embed, &provider.get(*b), cache, info);
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
                .map(|(_, assignments)| Self::centroid(provider, assignments, cache, info))
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

    fn remove(ixs: &mut Vec<usize>, index: usize) {
        ixs.retain(|&ix| ix != index);
    }

    fn build_level<'a, E, D, T, C, I>(
        provider: &'a E,
        cache: &mut C,
        info: &mut I,
        cur_root_ix: usize,
        cur_all_ixs: Vec<usize>,
        max_node_size: usize,
        pre_cluster: Option<usize>,
    ) -> Node
    where
        E: EmbeddingProvider<'a, D, T>,
        D: Distance<T> + Copy,
        T: 'a,
        C: Cache,
        I: Info,
    {
        let mut node = Node::new(cur_root_ix);
        let num_k = if max_node_size * max_node_size > cur_all_ixs.len() {
            ((cur_all_ixs.len() as f64).sqrt() as usize).max(1)
        } else {
            max_node_size
        };
        if num_k == 1 || cur_all_ixs.len() <= num_k {
            cur_all_ixs.iter().for_each(|&ix| {
                let mut cnode = Node::new(ix);
                cnode.compute_radius();
                node.add_child(cnode, provider, cache, info);
            });
        } else {
            // TODO pre_cluster makes things slower
            let init_centroids = match pre_cluster {
                Some(pre_cluster) => {
                    if cur_all_ixs.len() <= pre_cluster * num_k * 2 {
                        None
                    } else {
                        Some(
                            Self::kmedoid(
                                provider,
                                cur_all_ixs
                                    .iter()
                                    .take(pre_cluster * num_k)
                                    .map(|&cix| cix)
                                    .collect(),
                                None,
                                num_k,
                                cache,
                                info,
                            )
                            .into_iter()
                            .map(|(cix, _)| cix)
                            .collect(),
                        )
                    }
                }
                None => None,
            };
            Self::kmedoid(provider, cur_all_ixs, init_centroids, num_k, cache, info)
                .into_iter()
                .for_each(|(centroid_ix, mut assignments)| {
                    Self::remove(&mut assignments, centroid_ix);
                    let child_node = Self::build_level(
                        provider,
                        cache,
                        info,
                        centroid_ix,
                        assignments,
                        max_node_size,
                        pre_cluster,
                    );
                    node.add_child(child_node, provider, cache, info);
                });
        }
        node.compute_radius();
        node
    }
}

impl<'a, E, D, T> Tree<'a, E, D, T> for FannTree
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    T: 'a,
{
    fn build<C, I>(
        provider: &'a E,
        max_node_size: Option<usize>,
        pre_cluster: Option<usize>,
        cache: &mut C,
        info: &mut I,
    ) -> Self
    where
        C: Cache,
        I: Info,
    {
        let mut all_ixs: Vec<usize> = provider.all().collect();
        let max_node_size = match max_node_size {
            Some(max_node_size) => max_node_size,
            None => all_ixs.len(),
        };
        let root_ix = Self::centroid(provider, &all_ixs, cache, info);

        Self::remove(&mut all_ixs, root_ix);
        Self {
            root: Self::build_level(
                provider,
                cache,
                info,
                root_ix,
                all_ixs,
                max_node_size,
                pre_cluster,
            ),
            hash: provider.compute_hash(),
            distance_name: provider.distance().name().to_string(),
        }
    }

    fn draw<I>(
        &self,
        high_ix: usize,
        info: Option<&I>,
        res: Option<Vec<(usize, f64)>>,
        prune: bool,
        radius: bool,
    ) -> String
    where
        I: Info,
    {
        let pad = format!("{high_ix}", high_ix = high_ix).len();
        let show_ixs: HashMap<usize, bool> = {
            let mut show_ixs = HashMap::new();
            if let Some(info) = info {
                info.dist_vec().iter().for_each(|ix| {
                    show_ixs.insert(*ix, false);
                });
            }
            if let Some(res) = res {
                res.iter().for_each(|(ix, _)| {
                    show_ixs.insert(*ix, true);
                });
            }
            show_ixs
        };
        let stats: HashMap<usize, &str> = {
            let mut stats = HashMap::new();
            if let Some(info) = info {
                stats.extend(info.scan_map());
            }
            stats
        };
        self.root
            .draw(pad, &show_ixs, &stats, prune, radius)
            .join("\n")
    }

    fn get_closest<L, I>(
        &self,
        count: usize,
        provider: &'a E,
        cache: &mut L,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
        L: LocalCache<'a, D, T>,
    {
        // TODO caching makes things slower
        let mut res: Vec<(usize, DistanceCmp)> = Vec::with_capacity(count + 1);
        self.root
            .get_closest(&mut res, count, provider, cache, info);
        let distance = provider.distance();
        res.iter()
            .map(|(ix, v)| (*ix, distance.finalize_distance(v)))
            .collect()
    }

    fn fingerprint(&self) -> (&str, &str) {
        (&self.hash, &self.distance_name)
    }
}
