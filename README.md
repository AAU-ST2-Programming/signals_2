# Filtrering og mekaniske signaler

- **Lecture specific files**: files/* – `En mappe med filer til øvelser og eksempler fra undervisningen.`

---

## Forberedelse til lektionen

Følg denne guide nøje for at være klar til undervisningen:

### 1. Literatur

**Primær litteratur:**
- [Data Wrangling with Python af Jacek Gołębiewski (PDF)](https://datawranglingpy.gagolewski.com/datawranglingpy.pdf)
  - Kapitel 4.3: Inspecting the data distribution with histograms
- [Databeskyttelsesloven (Retsinformation)](https://www.retsinformation.dk/eli/lta/2018/502)
  - article. 5 (principper), 
  - article 9 (særlige kategorier af personoplysninger),
  - article. 28 (databehandler), 
  - article. 32 (sikkerhed), 
  - article. 35 (DPIA). 
**Supplerende litteratur:**
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)


---

### 2. Installationer og opsætning
- Sørg for at Python og VS Code er installeret (se evt. tidligere guides).
- Tjek at du har følgende extensions i Visual Studio Code:
  - `Python`
  - `jupyter`
- Download eller opdater materialet:

> ```zsh
> git clone https://github.com/AAU-ST2-Programming/signals_2.git
> cd signals_2
> git pull
> ```

---

# Mål for dagens forelæsning

- Forstå SCG/PCG signaler og deres frekvensindhold
- Finde S1 og S2 lyde
- Synkronisér events/features mellem ECG og SCG (R→S1,S2 timing)
- Kvantificér variabilitet med histogrammer

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
