use crate::memory::{self, Memory};

const STRICT_CHECKING: bool = false;

pub(crate) struct Bus {
    pub(crate) zero_memory: memory::ZeroMemory,

    pub(crate) sp_dmem: memory::SpDmem,
    pub(crate) sp_imem: memory::SpImem,
    pub(crate) rom: memory::Rom,
}

impl Bus {
    pub(crate) fn new() -> Self {
        Self {
            zero_memory: memory::ZeroMemory,
            sp_dmem: memory::SpDmem::new(),
            sp_imem: memory::SpImem::new(),
            rom: memory::Rom::new(),
        }
    }

    /// Converts the given virtual address to a physical address
    fn virtual_addr_to_physical_addr(&self, address: u32) -> u32 {
        match address {
            // KSEG0
            0x8000_0000..=0x9FFF_FFFF => address - 0x8000_0000,
            // KSEG1
            0xA000_0000..=0xBFFF_FFFF => address - 0xA000_0000,

            _ => panic!("unexpected virtual address {address:#010x}"),
        }
    }

    /// Converts the given physical address to a device, such as ROM
    fn physical_addr_to_device(&mut self, address: u32) -> Option<&mut dyn Memory> {
        match address {
            0x0400_0000..=0x0400_0FFF => Some(&mut self.sp_dmem),
            0x0400_1000..=0x0400_1FFF => Some(&mut self.sp_imem),
            0x0470_0000..=0x047F_FFFF if !STRICT_CHECKING => Some(&mut self.zero_memory),
            0x1000_0000..=0x1FBF_FFFF => Some(&mut self.rom),

            _ => panic!("unexpected physical address {address:#010x}"),
        }
    }
}

/// A macro that implements `Memory` getters and setters for the `Bus`
macro_rules! impl_mem_bus {
    ($get_name:ident, $set_name:ident, $type:ty) => {
        fn $get_name(&mut self, address: u32) -> $type {
            let physical_address = self.virtual_addr_to_physical_addr(address);
            self.physical_addr_to_device(physical_address)
                .unwrap_or_else(|| {
                    panic!("device at address {:#010x} should exist", physical_address)
                })
                .$get_name(physical_address)
        }

        fn $set_name(&mut self, address: u32, value: $type) {
            let physical_address = self.virtual_addr_to_physical_addr(address);
            if let Some(device) = self.physical_addr_to_device(physical_address) {
                device.$set_name(physical_address, value);
            }
        }
    };
}

impl Memory for Bus {
    impl_mem_bus!(get_u8, set_u8, u8);
    impl_mem_bus!(get_u32, set_u32, u32);
}
