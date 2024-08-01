# Coq

## Opam

```bash
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
Install in:

# Remember to add /opam_install to YOUR_PATH
export PATH="$PATH:YOUR_PATH/opam_install"

opam init --disable-sandboxing
```

## Initial setup (8.10.0)

```bash
opam switch create coq_8_10_0 4.07.1
opam pin add coq 8.10.0
opam install coq-serapi.8.10.0+0.7.0

# opam search -V serapi
# coq-serapi.8.10.0+0.7.0
# opam install coq.8.10.0
```

## Initial setup (8.11.0)

```bash
opam switch create coq_8_11_0 4.07.1
opam pin add coq 8.11.0
opam install coq-serapi.8.11.0+0.11.0

# opam search -V serapi
# coq-serapi.8.10.0+0.7.0
# opam install coq.8.10.0
```

## Libraries

Install bignums: https://github.com/coq/bignums

### Coq projects

#### algebra



### Retrieve all diffs (created for compatibility issues)

for NAME in $(ls -1 -d */ | cut -f1 -d'/'); do
  if [ "$NAME" == "PROCESSED" -o "$NAME" == "GITDIFFS" ]; then
    continue
  fi
  DIFF_PATH=$PWD/GITDIFFS/$NAME.diff
  cd $NAME
  git diff -- '*.v' > $DIFF_PATH
  cd ..
  if [ ! -s $DIFF_PATH ] ; then
    rm $DIFF_PATH
  else
    echo "Found diffs in $NAME"
  fi
done


## Manual SerAPI installation

```bash
# install Coq
opam switch create coq_8_11_0_manual_4081_2 4.08.1
opam pin add coq 8.11.0
opam install cmdliner sexplib dune ppx_import ppx_deriving ppx_sexp_conv yojson ppx_deriving_yojson

# clone remove_notations branch
cd resources/coq/vsiles_serapi
git clone https://github.com/vsiles/coq-serapi.git
cd coq-serapi/
git checkout remove_notations

# install SerAPI
make

# run
dune exec sertop
dune exec sertop --root resources/coq/vsiles_serapi/coq-serapi

(Add () "Unset Printing Notations.")

```
