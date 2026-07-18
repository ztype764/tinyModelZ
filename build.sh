#!/bin/bash
set -e

# Run the complete test suite using Maven
echo "Compiling and running test suite via Maven..."
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn test-compile exec:java -Dexec.mainClass="com.tinymodelz.TestRunner" -Dexec.classpathScope="test"

