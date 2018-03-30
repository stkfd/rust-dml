
#[derive(Abomonation, Clone)]
pub struct FetchSlice {
    pub start_index: usize,
    pub length: usize
}

impl FetchSlice {
    pub fn new(start_index: usize, length: usize) -> Self {
        FetchSlice { start_index, length }
    }
}

pub struct FetchChunks {
    pub chunk_length: usize
}

impl FetchChunks {
    pub fn new(chunk_length: usize) -> Self {
        FetchChunks { chunk_length }
    }
}
