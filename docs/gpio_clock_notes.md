# SKY130 GPIO ratings, internal timing and clocks

The SKY130 name identifies the process/I/O library, not a universal frequency
limit for all 130nm chips. Tiny Tapeout's [GPIO specification](https://tinytapeout.com/specs/gpio/)
records nominal SKY130 input-pad and output-pad ratings of 66MHz and 33MHz.
These concern pad/peripheral electrical behavior, edges and loading.
The specification notes ambiguity about output toggle versus square-wave
frequency, so these figures are not a guaranteed protocol MB/s budget.

Internal Fmax is a register-path timing property under specified PVT, clock,
uncertainty and extraction assumptions. An input-pad rating neither determines
nor proves the core's Fmax.

The external clock follows the input-pad and Tiny Tapeout clock-selection path
before reaching the macro. This design has no PLL, multiplier or separately
validated high-speed clock source. Slower data I/O alone does not create a
faster internal clock.

The qualified 64B/one-lane macro meets its 10ns/100MHz internal STA constraint.
That does not qualify 100MHz through the board/clock-pad path. A 50MHz initial
board experiment remains a proposed bring-up point, subject to complete
clock/I/O checks. ESL 5/25MB/s payload bandwidth values are separate assumptions.
Other Tiny Tapeout technologies require their own pad and board analysis.
See [physical evidence](tt_signoff_results.md).
