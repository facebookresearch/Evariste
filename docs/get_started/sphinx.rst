Writing documentation
=====================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documentation, and Github pages for deployment.
Documentation happens in two different places in this repo : 

* in the docs folder, as .rst files describing the overall codebase, referencing code documentation or writing tutorials

* in the code as docstrings

Once your documentation changes are on master, they must be *published* using github actions. 

Development
-----------
In order to test documentation changes, go to the ``Evariste/docs`` folder and run ``make html``.
All html will be produced in ``Evariste/docs/html``.

Run ``python3 -m http.server 8000`` in this folder to access the doc at ``http://localhost:8000``.

Deployment
----------
Deploy the docs using the github action defined in `.github/workflows/sphinx.yml`.
