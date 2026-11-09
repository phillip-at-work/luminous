
| Current State | Available Next States | Ray Splitting? | Shader Applied |
| :--- | :--- | :---: | :---: |
| **Start** | → Start Mask Check | No | - |
| **Start Mask Check** | → Reflect-Surface<br>→ **[Reflect-Surface + Transmit-In]**<br>→ Reflect-Interior<br>→ **[Reflect-Interior + Transmit-Out]**<br>→ End | No<br>**Yes**<br>No<br>**Yes**<br>No | Reflection<br>Reflection & Transmission (respectively)<br>Reflection<br>Reflection & Transmission (respectively)<br>Emission |
| **Reflect-Surface** | → Reflect-Surface<br>→ **[Reflect-Surface + Transmit-In]**<br>→ End | No<br>**Yes**<br>No | Reflection<br>Reflection & Transmission (respectively)<br>- |
| **Transmit-In** | → Reflect-Interior **(TIR)**<br>→ Transmit-Out **(Brewster's Angle)**<br>→ **[Reflect-Interior + Transmit-Out]**<br>→ End | No<br>No<br>**Yes**<br>No | Reflection<br>Transmission<br>Reflection & Transmission (respectively)<br>- |
| **Reflect-Interior** | → Reflect-Interior<br>→ **[Reflect-Interior + Transmit-Out]**<br>→ End | No<br>**Yes**<br>No | Reflection<br>Reflection & Transmission (respectively)<br>- |
| **Transmit-Out** | → Reflect-Surface **(TIR)**<br>→ Transmit-In **(Brewster's Angle)**<br>→ **[Reflect-Surface + Transmit-In]**<br>→ End | No<br>No<br>**Yes**<br>No | Reflection<br>Transmission<br>Reflection & Transmission (respectively)<br>- |
| **End** | → End Mask Check | No | - |
