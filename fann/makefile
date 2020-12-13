diff:
	clear && git status && git diff

commit:
	# e.g. `make commit f="source.py" m="commit message"`
	echo "files = $f" && echo "commit message = $m"
	git add $f && git commit -m "$m" && git push && git status
