echo ""
echo "1) Add \${CUDAVIEW}/contrib/slang/lib to LD_LIBRARY_PATH"
echo "     export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${CUDAVIEW}/contrib/slang/lib"
echo ""

echo "2) Make sure it was added: echo \$LD_LIBRARY_PATH"
echo "echo \$LD_LIBRARY_PATH"
echo ""

echo "3) Run from CUDAVIEW\'s main dir"
echo "Example:"
echo "    samples/CA3D-example/prog $((2**8)) 1 8 113 100 0.1 1"
