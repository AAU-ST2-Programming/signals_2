# Signal 2: Frekvensanalyse og FFT

- **Lecture specific files**: files/* – `En mappe med filer til øvelser og eksempler fra undervisningen.`

---

## Forberedelse til lektionen

Følg denne guide nøje for at være klar til undervisningen:

### 1. Literatur

**Primær litteratur:**
- [Think Python (online bog)](https://allendowney.github.io/ThinkPython/)
  - Kapitel 9: Lists (repetition)
- [Python for Everybody af Charles Severance (PDF)](https://do1.dr-chuck.com/pythonlearn/EN_us/pythonlearn.pdf)
  - Kapitel 8: Arbejde med arrays
- [Data Wrangling with Python af Jacek Gołębiewski (PDF)](https://datawranglingpy.gagolewski.com/datawranglingpy.pdf)
  - Kapitel 4.3: Inspecting the data distribution with histograms

- https://eur-lex.europa.eu/eli/reg/2016/679/oj  
  - article. 5 (principper), 
  - article 9 (særlige kategorier af personoplysninger),
  - article. 28 (databehandler), 
  - article. 32 (sikkerhed), 
  - article. 35 (DPIA). 
- https://www.retsinformation.dk/eli/lta/2018/502
  - Dansk lov: Databeskyttelsesloven, bl.a. §10 om behandling til statistiske/videnskabelige undersøgelser.  
**Supplerende litteratur:**
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)

**Formål:** Forstå Fourier-transformationer, FFT og frekvensanalyse af signaler.

---

### 2. Installationer og opsætning
- Sørg for at Python og VS Code er installeret (se evt. tidligere guides).
- Tjek at du har følgende extensions i Visual Studio Code:
  - `Python`
  - `jupyter`
- Download eller opdater materialet:
  ```bash
  cd ~/ST2-AnvendtProgrammering/signal_2
  git pull
  ```

---

## Lektionens fokus

- Fourier-transformationer og FFT
- Frekvensspectra og spektral analyse
- Filterdesign og filtrering
- Praktiske eksempler med EKG- og ECG-data

---

## Forventninger til forberedelse og undervisning

- **Før/efter kursusgang:**
  - Gennemgå tidligere kursusgange og kodeeksempler
  - Læs nyt materiale som beskrevet ovenfor
- **Tidsforbrug:**
  - 4 timers forberedelse (hjemme, før undervisning)
  - 4 timers undervisning og gruppeopgaver
  - 4 timers individuel opgaveregning (hjemme, efter undervisning)

---

## Spørgsmål og opgaver

- Til hver opgave i undervisningen vil der være:
  - En opgavebeskrivelse
  - En guide til hvordan opgaven løses
  - Svar på opgaven
- Opgaverne bygger videre på hinanden og bliver gradvist sværere.
- Til eksamen vil der kun være en opgavebeskrivelse – du skal selv kunne vurdere, hvordan opgaven løses.

---

**Husk:** Brug "Data Wrangling with Python" kapitel 7-8 som din primære kilde!