pub(crate) fn sign_extend_u16_to_u64(num: u16) -> u64 {
    ((num as i16) as i64) as u64
}

pub(crate) fn sign_extend_u32_to_u64(num: u32) -> u64 {
    ((num as i32) as i64) as u64
}

pub(crate) fn sign_extend_u16_to_i64(num: u16) -> i64 {
    (num as i16) as i64
}
