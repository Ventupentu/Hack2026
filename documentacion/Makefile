TARGET=main
LATEX=pdflatex
BIBTEX=bibtex
FLAGS=-interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean

all: pdf

pdf:
	$(LATEX) $(FLAGS) $(TARGET).tex
	$(BIBTEX) $(TARGET)
	$(LATEX) $(FLAGS) $(TARGET).tex
	$(LATEX) $(FLAGS) $(TARGET).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
