# Vectrex Development Reference

Authoritative local reference for the `vectrex-specialist` agent.
This file is the single source of truth for Vectrex-specific details (memory map, BIOS routines, ROM header, templates).

---

## 1. Hardware Overview & Memory Map

### CPU & Registers

* CPU: Motorola 6809, 8-bit accumulators, 16-bit index/stack registers
* Main registers:

  * A, B: 8-bit accumulators
  * D: 16-bit (A high, B low)
  * X, Y: 16-bit index registers
  * U, S: 16-bit stack pointers (user/system)
  * DP: 8-bit direct page (upper byte of 16-bit address in direct mode)
  * CC: condition code register
  * PC: program counter

### Memory Map

```
$0000–$7FFF  Cartridge ROM (up to 32 KB typical)
$8000–$BFFF  (Banks for larger carts; bank-switching if used)
$C800–$CBEA  RAM (approx. 1 KB total)
  $C800–$C87F  BIOS/system variables
  $C880–$CBEA  User RAM (~618 bytes usable)
$CBEA–$CBFF  System stack (grows downward)
$D000–$D01F  VIA 6522 (I/O, timers, vector control) – mirrored
$E000–$FFFF  BIOS ROM (8 KB)
```

### VIA 6522 (Key Ports)

* Base address: $D000 (mirrored in $D020, etc.)
* Important registers (using base $D000):

  * $D000: Port B (digital inputs/outputs, joystick buttons, etc.)
  * $D001: Port A (DAC output, multiplexer control)
  * $D002: DDRB
  * $D003: DDRA
  * $D004–$D007: Timer 1
  * $D008–$D009: Timer 2
  * $D00B: ACR (Aux control)
  * $D00C: PCR (Peripheral control)
  * $D00D: IFR (Interrupt flags)
  * $D00E: IER (Interrupt enable)

### Direct Page

* Convention: DP = $C8
* This makes direct addressing refer to $C800–$C8FF (fast access to system and user variables).
* Most BIOS routines assume DP = $C8.

---

## 2. Core Vectrex Rules

The agent must respect these rules in all code:

1. Call `Wait_Recal` ($F192) once per frame in the main loop.
2. Set DP to $C8 before using BIOS routines:

   * `LDA #$C8` / `TFR A,DP`
3. Use `Reset0Ref` ($F1AF) or `Wait_Recal` to recenter beam before drawing.
4. End all vector lists with `$01`.
5. Terminate all strings for BIOS printing with a byte whose high bit is set (e.g. `$80`, or last char OR $80).
6. Do not exceed user RAM ($C880–$CBEA) for custom variables.
7. Keep coordinates in the approximate Vectrex range: `-127..+127`.
8. ROM must start at `$0000` and include a valid header (copyright string `"g GCE YEAR"` + terminator).
9. ROM should be padded to a valid size (4 KB, 8 KB, 16 KB, 32 KB) with `$FF`.
10. Do NOT add CPU vector tables - the BIOS handles entry from the header.

---

## 3. ROM Header Format

### Canonical ROM Header (Assembly)

```asm
        ORG     $0000

        FCC     "g GCE 2025"    ; copyright string (g + space + GCE + space + year)
        FCB     $80             ; string terminator
        FDB     MusicData       ; pointer to music data
        FCB     $F8,$50,$20,$D0 ; height, width, rel Y, rel X for title display
        FCC     "GAME TITLE"    ; game title shown on startup
        FCB     $80             ; title terminator
        FCB     $00             ; header end marker

MusicData:
        FCB     $C8,$20,$00     ; minimal music header
        FCB     $01,$01,$01     ; music data
        FCB     $80             ; music end
```

### Key Header Details

* Copyright is a STRING `"g GCE YEAR"` not just byte `$67`
* Must include `$80` terminator after copyright string
* Display params are 4 separate bytes, not FDB words
* Header ends with `$00` byte
* Music data format requires the specific header bytes shown

### Important: No Vector Table Needed

Unlike standard 6809 systems, Vectrex cartridges do NOT need CPU vector tables:
* The Vectrex BIOS ROM contains the 6809 CPU vectors at `$FFF0-$FFFF`
* The BIOS reads the cartridge header and jumps to code immediately after it
* Do NOT place vectors at `$1FF0`, `$3FF0`, or `$7FF0` - this is incorrect

---

## 4. Minimal Assembly Template

This is the canonical "hello Vectrex" template used by the agent.

```asm
        ORG     $0000

        FCC     "g GCE 2025"
        FCB     $80
        FDB     MusicData
        FCB     $F8,$50,$20,$D0
        FCC     "HELLO WORLD"
        FCB     $80
        FCB     $00

MusicData:
        FCB     $C8,$20,$00
        FCB     $01,$01,$01
        FCB     $80

ColdStart:
        LDA     #$C8
        TFR     A,DP
        LDS     #$CBEA
        LDU     #$C880

MainLoop:
        JSR     $F192           ; Wait_Recal

        LDA     #$7F
        JSR     $F2AB           ; Intensity_a

        LDA     #$30            ; Y position
        LDB     #$E0            ; X position
        LDX     #TextHello
        JSR     $F495           ; Print_Str_d

        BRA     MainLoop

TextHello:
        FCC     "HELLO WORLD"
        FCB     $80

        END
```

Note: No vector table at end of ROM - the BIOS handles entry point detection from header.

---

## 5. BIOS Routines (Core Subset)

Addresses are in the $E000–$FFFF BIOS region. This list is the primary reference.

### Frame & Beam

```text
$F000  Cold_Start    ; Full BIOS init (called once on boot/reset)
$F192  Wait_Recal    ; Wait for frame sync + reset integrators
$F1AA  DP_to_C8      ; Set DP to $C8
$F1AF  Reset0Ref     ; Reset integrators (center beam)
```

### Intensity & Vector Drawing

```text
$F2AB  Intensity_a   ; A = intensity (0–$7F)
$F2B5  Intensity_5F  ; Medium intensity
$F2BC  Intensity_7F  ; Max intensity

$F373  Moveto_d      ; A=Y, B=X – absolute move
$F40E  Draw_Line_d   ; A=ΔY, B=ΔX – draw line from current pos

$F3CE  Draw_VL       ; X=list, Y=position, default scale
$F3AD  Draw_VL_a     ; X=list, Y=position, A=scale
$F389  Draw_VL_mode  ; X=list, Y=position, A=scale, B=mode
$F312  Dot_ix_b      ; X=position, B=intensity or small delta (implementation-specific)
$F35B  Dot_List      ; X=dot list (pairs), $01 terminator
```

### Text

```text
$F495  Print_Str_d   ; A=Y, B=X, X=string
$F4A2  Print_Str_yx  ; X=string, Y=position
$F4B0  Print_List    ; X=list of string pointers
$F543  Print_Ships_x ; A=number, X=position (formatted)
```

### Input

```text
$F1F5  Read_Btns     ; Updates button states at $C81B-$C81E
$F1F8  Read_Btns_Mask; A=mask in, A=masked result out
$F1BA  Joy_Analog    ; Reads analog joystick, results in $C819/$C81A
```

Typical BIOS joystick variables (after Joy_Analog):

```text
$C819  Joystick Y (signed, -127..+127)
$C81A  Joystick X (signed, -127..+127)
$C81B  Joystick 1 buttons (bits 0–3)
```

### Randomness

```text
$F603  Random        ; A = random 8-bit
$F610  Random_3      ; A = random 0–7
```

### Sound/Music (High-level)

```text
$F272  Do_Sound      ; X=pointer to sound data, B=duration
$F27D  Init_Music_chk; Initialize music if not already
$F284  Init_Music    ; X=pointer to music data
$F28C  Do_Sound_x    ; X=pointer to sound data
```

---

## 6. Vector Lists

### Format

Typical list used with `Draw_VL_a` / `Draw_VL`:

```asm
; First byte: mode (usually $FF = draw)
; Optional second "submode" byte (often also $FF)
; Then pairs: Y, X offsets (signed bytes)
; End of list: $01

ShipVectors:
        FCB     $FF,$FF
        FCB     $00,$20         ; right
        FCB     $20,$00         ; down
        FCB     $00,$E0         ; left
        FCB     $E0,$00         ; up
        FCB     $01
```

### Drawing a Vector List

```asm
DrawShip:
        LDX     #ShipVectors
        LDD     PlayerPos       ; 16-bit packed pos, e.g. high=Y, low=X
        TFR     D,Y             ; Y = position
        LDA     #$60            ; scale
        JSR     $F3AD           ; Draw_VL_a
        RTS
```

### Common Conventions

* Scale: `A` in `Draw_VL_a` is a 0–255-ish scaler; smaller → smaller vectors.
* Order: draw background (dim intensity) → gameplay objects → UI.

---

## 7. Minimal CMOC Template

Standard starting point for C-based projects.

```c
#include <vectrex.h>
#include <vectrex/bios.h>

int main(void) {
    while (1) {
        wait_retrace();              // Wait_Recal
        intensity(0x7F);             // max brightness
        print_str_d(40, 0, "HELLO VECTREX");
    }
    return 0;
}
```

### Typical CMOC Build Command

* Source layout:

```
src/main.c
build/game.vec
```

* Command:

```sh
cmoc --vectrex -o build/game.vec src/main.c
```

### Emulator Example (MAME)

```sh
mame vectrex -cart build/game.vec
```

Adjust emulator path and options as needed.

---

## 8. Input Handling Patterns

### Simple Analog Read (Assembly)

```asm
ReadInput:
        JSR     $F1BA           ; Joy_Analog
        JSR     $F1F5           ; Read_Btns

        ; After this:
        ; $C819 = Y, $C81A = X, $C81B = buttons

        RTS
```

### 8-Way Direction Decode

```asm
; Output: A = 0 (center) or 1..8 for directions
GetDirection:
        CLRA

        LDB     $C819           ; Y
        CMPB    #40
        BLT     CheckDown
        LDA     #1              ; up

CheckDown:
        CMPB    #-40
        BGT     CheckRight
        LDA     #5              ; down

CheckRight:
        LDB     $C81A
        CMPB    #40
        BLT     CheckLeft

        CMPA    #1
        BEQ     UpRight
        CMPA    #5
        BEQ     DownRight
        LDA     #3              ; right
        BRA     Done

UpRight: LDA     #2
         BRA     Done
DownRight:
         LDA     #4
         BRA     Done

CheckLeft:
        CMPB    #-40
        BGT     Done

        CMPA    #1
        BEQ     UpLeft
        CMPA    #5
        BEQ     DownLeft
        LDA     #7              ; left
        BRA     Done

UpLeft:  LDA     #8
         BRA     Done
DownLeft:
         LDA     #6

Done:
        RTS
```

---

## 9. AY-3-8912 PSG (Short Reference)

Access is indirect through VIA; exact wiring is hardware-specific and often hidden behind BIOS routines. For most use-cases, prefer BIOS sound/music routines unless you need full manual control.

### Strategy

* For simple games: use `Init_Music`, `Do_Sound`, and prebuilt sound/music tables.
* For advanced control: maintain a small, dedicated sound driver that:

  * writes to PSG registers via VIA
  * manages envelope, noise, and channel mixing
  * updates once per frame or more frequently via timers

Because PSG wiring details are error-prone, the agent should:

* Prefer BIOS sound routines for general tasks.
* Only output low-level PSG write routines when necessary and keep them minimal and consistent.

---

## 10. Tooling & Build Conventions

### Assemblers

Preferred modern options:

* `lwasm` (part of lwtools)
* `asm6809`

### Build Process (Assembly)

**Important**: lwasm `--format=raw` does NOT pad ROMs. ROMs must be padded to valid sizes (4KB, 8KB, 16KB, 32KB).

```sh
# Assemble to raw binary
lwasm --format=raw --output=build/hello.raw src/hello.asm

# Create 4KB file filled with $FF
perl -e 'print "\xff" x 4096' > build/hello.vec

# Overlay raw code at start
dd if=build/hello.raw of=build/hello.vec conv=notrunc 2>/dev/null
```

### CMOC

* Install via package manager (e.g. Homebrew) when available.
* Build:

```sh
cmoc --vectrex -o build/game.vec src/main.c
```

### Makefile Template (Assembly)

```make
ROM = build/hello.vec
SRC = src/hello.asm

all: rom

rom: $(SRC)
	@mkdir -p build
	lwasm --format=raw --output=build/hello.raw $(SRC)
	perl -e 'print "\xff" x 4096' > $(ROM)
	dd if=build/hello.raw of=$(ROM) conv=notrunc 2>/dev/null
	@rm build/hello.raw
	@echo "ROM built: $(ROM)"
	@ls -lh $(ROM)

run: rom
	mame vectrex -rompath ~/.mame/roms -cart $(ROM) -window

clean:
	rm -rf build

.PHONY: all rom run clean
```

---

## 11. Emulator Notes

### MAME Setup

MAME requires the Vectrex BIOS ROM to run cartridges.

**Required File:**
* `exec_rom.bin` (8KB Vectrex executive ROM)

**Installation:**
```sh
mkdir -p ~/.mame/roms/vectrex
cp /path/to/exec_rom.bin ~/.mame/roms/vectrex/
```

**Running:**
```sh
mame vectrex -rompath ~/.mame/roms -cart build/game.vec -window
```

**Useful Options:**
```sh
mame vectrex -rompath ~/.mame/roms -cart build/game.vec -window -nomax
mame vectrex -rompath ~/.mame/roms -cart build/game.vec -debug
```

### OpenEmu

OpenEmu uses libretro cores for Vectrex emulation.
* Expects `.vec` file extension
* Same ROM format as MAME
* No separate BIOS setup required (built into core)
* If ROM shows Mine Storm instead of your game, the header is invalid

### ParaJVE / VecX

* Use if available; specifics depend on current distribution.
* Agent should not hard-code installation paths; show generic examples and let the user adapt.

---

## 12. Troubleshooting Checklist

Display issues:

* Black screen:

  * Is the ROM header valid? (Must start with `"g GCE YEAR"` string, not just `$67`)
  * Is `Wait_Recal` called in the main loop?
  * Is intensity > 0 before drawing?
* Off-center or drifting:

  * `Wait_Recal` and/or `Reset0Ref` missing or misused.
* Flicker:

  * Too many vectors per frame; reduce complexity or split across frames.
* Shows Mine Storm instead of your game:

  * Header is invalid - emulator falls back to built-in game.
  * Check copyright string format and terminators.

Logic issues:

* No input:

  * `Joy_Analog` / `Read_Btns` not called.
  * Using wrong addresses for $C819/$C81A/$C81B.
* Stuck loop:

  * Missing `Wait_Recal`, tight busy-waiting loops.

Build issues:

* ROM won't boot:

  * Header format incorrect (see Section 3).
  * ROM not padded to valid size (4KB, 8KB, 16KB, 32KB).
  * Using `FILL` directive instead of external padding.
* Branch out of range:

  * Use long branches (`LBRA`, `LBSR`) for distant targets.

Performance:

* Slow framerate:

  * Too many long vectors or complex loops per frame.
  * Unnecessary `PSHS/PULS` in inner loops.

The agent should use this document as the canonical reference and extend with small, focused ROMs and examples instead of embedding giant templates into every change.
