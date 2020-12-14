diff:
	clear && git status && git diff

commit:
	# e.g. `make commit f="source.py" m="commit message"`
	# TODO(sparshsah): why can't i seem to do `f=${f:="."}` or `[[ -z "$f" ]] && f="."`
	echo "files = $f" && echo "commit message = $m"
	git add $f && git commit -m "$m" && git push && git status
