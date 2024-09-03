use std::{fs, io, path::Path};

pub(crate) trait Memory {
    /// Read a `u8` from the given address
    fn get_u8(&mut self, address: u32) -> u8;
    /// Write a `u8` to the given address
    fn set_u8(&mut self, address: u32, value: u8);

    /// Read a `u32` from the given address
    fn get_u32(&mut self, address: u32) -> u32;
    /// Write a `u32` to the given address
    fn set_u32(&mut self, address: u32, value: u32);
}

/// A struct that implements the `Memory` trait and always returns zero
pub(crate) struct ZeroMemory;

impl Memory for ZeroMemory {
    fn get_u8(&mut self, _address: u32) -> u8 {
        0
    }

    fn set_u8(&mut self, _address: u32, _value: u8) {
        // No operation
    }

    fn get_u32(&mut self, _address: u32) -> u32 {
        0
    }

    fn set_u32(&mut self, _address: u32, _value: u32) {
        // No operation
    }
}

/// A wrapper around `Vec` that implements the `Memory` trait
pub(crate) struct VecMemory<const O: u32, const S: usize> {
    bytes: Vec<u8>,
}

impl<const O: u32, const S: usize> VecMemory<O, S> {
    pub(crate) fn new() -> Self {
        Self { bytes: vec![0; S] }
    }
}

impl<const O: u32, const S: usize> Memory for VecMemory<O, S> {
    fn get_u8(&mut self, address: u32) -> u8 {
        self.bytes[(address - O) as usize]
    }

    fn set_u8(&mut self, address: u32, value: u8) {
        self.bytes[(address - O) as usize] = value;
    }

    fn get_u32(&mut self, address: u32) -> u32 {
        let index = (address - O) as usize;
        u32::from_be_bytes(
            self.bytes[index..index + 4]
                .try_into()
                .expect("array should have length 4"),
        )
    }

    fn set_u32(&mut self, address: u32, value: u32) {
        let bytes = u32::to_be_bytes(value);
        for i in 0..4 {
            self.set_u8(address + i as u32, bytes[i]);
        }
    }
}

/// A macro for defining a `VecMemory`
macro_rules! vec_memory {
    ($start:expr => $end:expr) => {
        VecMemory<$start, { $end - $start + 1 }>
    };
}

pub(crate) type SpDmem = vec_memory!(0x0400_0000 => 0x0400_0FFF);
pub(crate) type SpImem = vec_memory!(0x0400_1000 => 0x0400_1FFF);
pub(crate) type Rom = vec_memory!(0x1000_0000 => 0x1FBF_FFFF);

impl Rom {
    /// Loads the contents of a ROM file
    pub(crate) fn load(&mut self, fp: impl AsRef<Path>) -> io::Result<()> {
        self.bytes = fs::read(fp)?;
        Ok(())
    }
}
