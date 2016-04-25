train:
	python ./training.py

test:
	python ./testing.py

doc:
	pdflatex -shell-escape ./doc/

clean:
	find ./ -name "*.pyc" -delete
	mv ./features/* ~/.trash/
	mv ./model/model.cache ~/.trash/

cleanPyc:
	find ./ -name "*.pyc" -delete

rmPyc:
	mv ./*.pyc ~/.trash/
