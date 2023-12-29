if [ -d "$VIRTUALENV_NAMESPACE" ]; then
    echo "$VIRTUALENV_NAMESPACE exists."
    source $VIRTUALENV_NAMESPACE/bin/activate
else
    make
    source $VIRTUALENV_NAMESPACE/bin/activate
fi