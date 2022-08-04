# Compilation of samples


## RISC-V

I used the following GCC compilers:
  * Native Ubuntu riscv64 compiler (OS V20, GCC V9.4)
  * riscv32 from [[https://github.com/stnolting/riscv-gcc-prebuilt]]

To prevent RCV instructions (with riscv64), a dedicated prolog is used
for standalone executable with the following options:
	start.s -nostartfiles -march=rv64g -static -g3
