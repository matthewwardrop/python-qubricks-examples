#!/bin/bash
cd `dirname $0`

pip_modules="cython numpy scipy sympy matplotlib"
git_modules="python-parameters python-qubricks python-mplstyles"

black='\033[30;47m'
red='\033[31;47m'
green='\033[32;47m'
yellow='\033[33;47m'
blue='\033[34;47m'
magenta='\033[35;47m'
cyan='\033[36;47m'
white='\033[37;47m'

# -----------------------

cecho ()                     # Color-echo.
                             # Argument $1 = message
                             # Argument $2 = color
{
local default_msg="No message passed."
                             # Doesn't really need to be a local variable.

message=${1:-$default_msg}   # Defaults to default message.
color=${2:-$black}           # Defaults to black, if not specified.

  echo -ne $color
  echo -n "$message"
  tput sgr0                      # Reset to normal.
  echo

  return
}  
#------------------------

if [[ `uname` == 'Linux' ]]; then
   echo "WARNING: On Linux systems, it is probably better to install the dependencies using your package manager. Continue? (y/N)" $red
   read response
   if [[ $response != 'y' ]]; then
	exit
   fi
fi


for module in $pip_modules; do
	cecho "Installing and/or checking for updates to: ${module}" $blue
	sudo pip2 install --upgrade ${module}
done

for module in $git_modules; do
	cecho "Cloning and/or updating git module: ${module}" $blue
	if [ ! -d $module ]; then
		echo git clone https://github.com/matthewwardrop/${module}.git
	fi
	pushd $module > /dev/null
	git pull
	sudo python2 setup.py install
	popd > /dev/null
done

cecho "Cloning and/or updating git module: python-qubricks-examples" $blue
if [ ! -d python-qubricks-examples ]; then
	git clone https://github.com/matthewwardrop/python-qubricks-examples.git
fi
pushd python-qubricks-examples > /dev/null
git pull
popd > /dev/null
