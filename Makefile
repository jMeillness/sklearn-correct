test:
	@echo
	@coverage run -m unittest -v sklc.test
	@echo
	@coverage report -m --include "sklc/*" --omit "sklc/test/*,**/__init__.py"
	@echo
	@coverage html --include "sklc/*" --omit "sklc/test/*,**/__init__.py" -d build/htmlcov
