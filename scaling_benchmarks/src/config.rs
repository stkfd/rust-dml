use std::fs::File;
use std::io::Read;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub classification_trees: ClassificationTrees,
    pub regression_trees: RegressionTrees,
    pub boosted_trees: BoostedTrees,
    pub kmeans: Kmeans,
}

pub fn read() -> Config {
    let mut config_file = File::open("config.toml").expect("Config file not found");
    let mut contents = String::new();
    config_file
        .read_to_string(&mut contents)
        .expect("something went wrong reading the config file");
    ::toml::from_str(&contents).expect("Config file format contained errors")
}

#[derive(Deserialize, Debug, Clone)]
pub struct ClassificationTrees {
    pub samples: usize,
    pub threads: Vec<usize>,
    pub levels: usize,
    pub bins: Vec<usize>,
}
#[derive(Deserialize, Debug, Clone)]
pub struct RegressionTrees {
    pub samples: usize,
    pub threads: Vec<usize>,
    pub levels: usize,
    pub bins: Vec<usize>,
    pub quantize_resolution: usize,
}
#[derive(Deserialize, Debug, Clone)]
pub struct BoostedTrees {
    pub threads: Vec<usize>,
    pub samples: usize,
    pub stages: Vec<usize>,
    pub levels: usize,
    pub bins: usize,
    pub quantize_resolution: usize,
    pub learning_rate: f64,
}
#[derive(Deserialize, Debug, Clone)]
pub struct Kmeans {
    pub samples: usize,
    pub threads: Vec<usize>,
}