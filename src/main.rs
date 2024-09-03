mod bit;
mod bus;
mod memory;

use std::{
    io,
    ops::{Index, IndexMut},
    path::Path,
};

use num_derive::FromPrimitive;

use bus::Bus;
use memory::Memory;

/// MIPS opcodes
#[derive(Debug, FromPrimitive)]
enum Opcode {
    Special = 0b000000,
    Bne = 0b000101,
    Addi = 0b001000,
    Addiu = 0b001001,
    Ori = 0b001101,
    Lui = 0b001111,
    Mtc0 = 0b010000,
    Lw = 0b100011,
    Sw = 0b101011,
}

impl From<u32> for Opcode {
    fn from(value: u32) -> Self {
        let opcode = value >> 26;
        num::FromPrimitive::from_u32(opcode).unwrap_or_else(|| {
            panic!("Unknown opcode {opcode:#08b}");
        })
    }
}

/// Operands for MIPS J-instructions
#[derive(Debug)]
struct InstructionOperandsJump {
    target: u32,
}

/// Operands for MIPS I-instructions
#[derive(Debug)]
struct InstructionOperandsImmediate {
    rs: usize,
    rt: usize,
    immediate: u16,
}

/// Operands for MIPS R-instructions
#[derive(Debug)]
struct InstructionOperandsRegister {
    rs: usize,
    rt: usize,
    rd: usize,
    sa: u8,
    funct: u8,
}

/// Operands for all types of MIPS instructions
#[derive(Debug)]
enum InstructionOperands {
    Jump(InstructionOperandsJump),
    Immediate(InstructionOperandsImmediate),
    Register(InstructionOperandsRegister),
}

/// A MIPS instruction
#[derive(Debug)]
struct Instruction {
    opcode: Opcode,
    operands: InstructionOperands,
}

impl From<u32> for Instruction {
    fn from(value: u32) -> Self {
        let opcode = value.into();

        Self {
            operands: match opcode {
                Opcode::Addi
                | Opcode::Addiu
                | Opcode::Bne
                | Opcode::Lui
                | Opcode::Lw
                | Opcode::Ori
                | Opcode::Sw => InstructionOperands::Immediate(InstructionOperandsImmediate {
                    rs: ((value >> 21) & 0b11111) as usize,
                    rt: ((value >> 16) & 0b11111) as usize,
                    immediate: (value & 0xFFFF) as u16,
                }),

                Opcode::Special | Opcode::Mtc0 => {
                    InstructionOperands::Register(InstructionOperandsRegister {
                        rs: ((value >> 21) & 0b11111) as usize,
                        rt: ((value >> 16) & 0b11111) as usize,
                        rd: ((value >> 11) & 0b11111) as usize,
                        sa: ((value >> 6) & 0b11111) as u8,
                        funct: (value & 0b111111) as u8,
                    })
                }
            },

            opcode,
        }
    }
}

/// The COP0 Cause register fields
#[derive(Debug)]
struct CauseRegister {
    branch_delay: bool,
    coprocessor_error: u8,
    interrupt_pending: [bool; 8],
    exception_code: u8,
}

impl From<CauseRegister> for u32 {
    fn from(val: CauseRegister) -> Self {
        let mut result = (val.branch_delay as u32) << 31
            | (val.coprocessor_error as u32) << 28
            | (val.exception_code as u32) << 2;

        for (i, bit) in val.interrupt_pending.iter().enumerate() {
            if *bit {
                result |= 1 << (i + 8);
            }
        }

        result
    }
}

impl From<u32> for CauseRegister {
    fn from(val: u32) -> Self {
        // Check that all zero areas are set to all zero
        assert_eq!((val >> 30) & 1, 0);
        assert_eq!((val >> 16) & 0b1111_1111_1111, 0);
        assert_eq!((val >> 7) & 1, 0);
        assert_eq!(val & 0b11, 0);

        let mut interrupt_pending = [false; 8];

        for (i, bit) in interrupt_pending.iter_mut().enumerate() {
            *bit = (val >> (i + 8)) & 1 == 1;
        }

        Self {
            branch_delay: (val >> 31) == 1,
            coprocessor_error: ((val >> 28) & 0b11) as u8,
            interrupt_pending,
            exception_code: ((val >> 2) & 0b11111) as u8,
        }
    }
}

/// MIPS CPU modes
enum CpuMode {
    User = 0b10,
    Supervisor = 0b01,
    Kernel = 0b00,
}

/// The COP0 Status register fields
struct StatusRegister {
    coprocessor_usable: [bool; 4],
    reduced_power: bool,
    additional_floating_point_registers: bool,
    reverse_endian: bool,
    instruction_trace_support: bool,
    boot_exception_vector_is_bootstrap: bool,
    tlb_shutdown: bool,
    soft_reset: bool,
    cp0_condition: bool,
    interrupt_mask: [bool; 8],
    supervisor_extended_addressing_enabled: bool,
    kernel_extended_addressing_enabled: bool,
    user_extended_addressing_enabled: bool,
    cpu_mode: CpuMode,
    is_handling_error: bool,
    is_handling_exception: bool,
    interrupts_enabled: bool,
}

/// All COP0 registers
#[derive(FromPrimitive)]
enum Cop0Register {
    Random = 1,
    Wired = 6,
    Count = 9,
    Compare = 11,
    Status = 12,
    Cause = 13,
    PrId = 15,
    Config = 16,
}

impl From<usize> for Cop0Register {
    fn from(value: usize) -> Self {
        num::FromPrimitive::from_usize(value).unwrap_or_else(|| {
            panic!("Unknown COP0 register: {value}");
        })
    }
}

/// Co-processor 0
struct Cop0 {
    registers: [u64; 32],
}

impl Cop0 {
    fn new() -> Self {
        Self { registers: [0; 32] }
    }

    /// Reads from a register and returns the result
    fn read(&self, register: Cop0Register) -> u64 {
        match register {
            Cop0Register::Random => self[register],
            Cop0Register::Count => self[Cop0Register::Count] >> 1,

            _ => todo!(),
        }
    }

    /// Writes to a register
    fn write(&mut self, register: Cop0Register, value: u64) {
        match register {
            Cop0Register::Wired => {
                self[Cop0Register::Wired] = value;
                self[Cop0Register::Random] = 0x1F;
            }

            Cop0Register::Count => {
                // Shift left one because the actual stored value is
                self[Cop0Register::Count] = value << 1;
            }

            Cop0Register::Cause => {
                let new_cause_register: CauseRegister = (value as u32).into();
                let mut current_cause_register: CauseRegister =
                    (self[Cop0Register::Cause] as u32).into();

                current_cause_register.interrupt_pending[0] =
                    new_cause_register.interrupt_pending[0];
                current_cause_register.interrupt_pending[1] =
                    new_cause_register.interrupt_pending[1];

                if new_cause_register.interrupt_pending[0]
                    || new_cause_register.interrupt_pending[1]
                {
                    panic!("Unhandled software interrupt");
                }

                let new_value: u32 = current_cause_register.into();
                self[Cop0Register::Cause] = new_value as u64;
            }

            Cop0Register::Compare => {
                self[register] = value;
            }

            _ => todo!(),
        }
    }
}

impl Index<Cop0Register> for Cop0 {
    type Output = u64;

    /// Raw register access! You should probably use `read` instead
    fn index(&self, index: Cop0Register) -> &Self::Output {
        &self.registers[index as usize]
    }
}

impl IndexMut<Cop0Register> for Cop0 {
    /// Raw register access! You should probably use `write` instead
    fn index_mut(&mut self, index: Cop0Register) -> &mut Self::Output {
        &mut self.registers[index as usize]
    }
}

/// The MIPS CPU
struct Cpu {
    program_counter: u64,

    // Used to emulate branch delay slot
    next_jump_address: Option<u64>,

    registers: [u64; 32],
    cop0: Cop0,
}

impl Cpu {
    fn new() -> Self {
        Self {
            program_counter: 0xBFC0_0000,
            next_jump_address: None,
            registers: [0; 32],
            cop0: Cop0::new(),
        }
    }

    /// Simulates the PIF ROM
    fn pif_setup(&mut self, bus: &mut Bus) {
        // Register setup

        self.registers[11] = 0xFFFF_FFFF_A400_0040;
        self.registers[20] = 0x0000_0000_0000_0001;
        self.registers[22] = 0x0000_0000_0000_003F;
        self.registers[29] = 0xFFFF_FFFF_A400_1FF0;

        // Is not normally written to directly in normal program execution
        self.cop0[Cop0Register::Random] = 0x1F;
        self.cop0[Cop0Register::Status] = 0x3400_0000;
        self.cop0[Cop0Register::PrId] = 0x0000_0B00;
        self.cop0[Cop0Register::Config] = 0x0006_E463;

        // Copy ROM into SP DMEM
        for i in 0..0x1000 {
            let byte = bus.get_u8(0xB000_0000 + i);
            bus.set_u8(0xA400_0000 + i, byte);
        }

        // Let's get this show on the road!
        self.program_counter = 0xA400_0040;
    }

    /// Simulates a single CPU tick
    fn tick(&mut self, bus: &mut Bus) {
        println!("{:#010x}", self.program_counter);

        let instruction_u32 = bus.get_u32(self.program_counter as u32);
        self.program_counter += 4;

        let next_program_counter = match self.next_jump_address {
            Some(address) => {
                self.next_jump_address = None;
                address
            }
            None => self.program_counter,
        };

        let instruction = instruction_u32.into();

        match instruction {
            Instruction {
                opcode: Opcode::Addi,
                operands: InstructionOperands::Immediate(operands),
            } => {
                // TODO: Handle overflow exception
                match (self.registers[operands.rs] as i64)
                    .checked_add(bit::sign_extend_u16_to_i64(operands.immediate))
                {
                    Some(value) => self.registers[operands.rt] = value as u64,
                    None => todo!(),
                }
            }

            Instruction {
                opcode: Opcode::Addiu,
                operands: InstructionOperands::Immediate(operands),
            } => {
                self.registers[operands.rt] = self.registers[operands.rs]
                    .wrapping_add(bit::sign_extend_u16_to_u64(operands.immediate));
            }

            Instruction {
                opcode: Opcode::Bne,
                operands: InstructionOperands::Immediate(operands),
            } => {
                if self.registers[operands.rs] != self.registers[operands.rt] {
                    self.next_jump_address = Some(
                        self.program_counter
                            .wrapping_add(bit::sign_extend_u16_to_u64(operands.immediate) << 2),
                    );
                }
            }

            Instruction {
                opcode: Opcode::Lui,
                operands: InstructionOperands::Immediate(operands),
            } => {
                self.registers[operands.rt] = (operands.immediate as u64) << 16;
            }

            Instruction {
                opcode: Opcode::Lw,
                operands: InstructionOperands::Immediate(operands),
            } => {
                let address = self.registers[operands.rs]
                    .wrapping_add(bit::sign_extend_u16_to_u64(operands.immediate));

                // TODO: Handle exception caused when address is not aligned
                assert_eq!(address & 0b11, 0);

                self.registers[operands.rt] = bus.get_u32(address as u32) as u64;
            }

            Instruction {
                opcode: Opcode::Mtc0,
                operands: InstructionOperands::Register(operands),
            } => {
                self.cop0
                    .write(operands.rd.into(), self.registers[operands.rt]);
            }

            Instruction {
                opcode: Opcode::Ori,
                operands: InstructionOperands::Immediate(operands),
            } => {
                self.registers[operands.rt] =
                    self.registers[operands.rs] | (operands.immediate as u64);
            }

            Instruction {
                opcode: Opcode::Special,
                operands: InstructionOperands::Register(operands),
            } => {
                if instruction_u32 == 0 {
                    // NOP. Let's do nothing.
                } else {
                    todo!()
                }
            }

            Instruction {
                opcode: Opcode::Sw,
                operands: InstructionOperands::Immediate(operands),
            } => {
                let address = self.registers[operands.rs]
                    .wrapping_add(bit::sign_extend_u16_to_u64(operands.immediate));

                // TODO: Handle exception caused when address is not aligned
                assert_eq!(address & 0b11, 0);

                bus.set_u32(address as u32, self.registers[operands.rt] as u32);
            }

            _ => panic!("Instruction not handled: {instruction:?}"),
        }

        self.cop0[Cop0Register::Count] += 1;

        if self.cop0[Cop0Register::Count] == self.cop0[Cop0Register::Compare] {
            // TODO: Handle interrupt here
            todo!();
        }

        self.cop0[Cop0Register::Random] = self.cop0[Cop0Register::Random].wrapping_sub(1);
        if self.cop0[Cop0Register::Random] < self.cop0[Cop0Register::Wired] {
            self.cop0[Cop0Register::Random] = 0x1F;
        }

        self.program_counter = next_program_counter;
    }
}

/// The Nintendo 64
struct N64 {
    cpu: Cpu,
    bus: Bus,
}

impl N64 {
    fn new() -> Self {
        Self {
            cpu: Cpu::new(),
            bus: Bus::new(),
        }
    }

    /// Creates a new `N64` with a ROM file preloaded
    fn with_rom(fp: impl AsRef<Path>) -> io::Result<Self> {
        let mut n64 = Self::new();
        n64.load_rom(fp)?;
        Ok(n64)
    }

    /// Loads a ROM file
    fn load_rom(&mut self, fp: impl AsRef<Path>) -> io::Result<()> {
        self.bus.rom.load(fp)
    }

    /// Starts the N64
    fn start(&mut self) {
        self.cpu.pif_setup(&mut self.bus);

        loop {
            self.tick();
        }
    }

    /// Simulates a single N64 clock tick
    fn tick(&mut self) {
        self.cpu.tick(&mut self.bus);
    }
}

fn main() {
    let mut n64 = N64::with_rom("./roms/sm64.z64").expect("rom file should exist");
    n64.start();
}
