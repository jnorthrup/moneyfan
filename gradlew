#!/usr/bin/env sh

#
# Copyright 2007-2022 the original author or authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This script is intended for use with the Gradle wrapper.
# It is a thin wrapper around the gradle-wrapper.jar file that ships with
# the Gradle wrapper.
#
# This script is not intended to be run directly. Instead, you should use
# the gradlew command in your project's root directory.
#
# For more information about the Gradle wrapper, please see
# https://docs.gradle.org/current/userguide/gradle_wrapper.html
#

set -e

# Determine the location of this script.
# Note that this script is not necessarily located in the project root.
# It could be located in a subdirectory of the project root, or it could
# be located in a completely different directory.
#
# The only assumption that this script makes is that the gradle-wrapper.jar
# file is located in the same directory as this script, or in a
# gradle/wrapper subdirectory of the directory that contains this script.
#
# The following logic is used to determine the location of this script:
# 1. If the BASH_SOURCE variable is set, then use it.
# 2. Otherwise, if the _ variable is set, then use it.
# 3. Otherwise, use $0.
#
# Note that the BASH_SOURCE variable is only set in Bash, and the _
# variable is only set in some shells (e.g. Bash and Zsh).
#
# Also note that the $0 variable is not always reliable. For example,
# if this script is executed by a symlink, then $0 will be the path to
# the symlink, not the path to this script.
#
# For these reasons, the following logic is used to determine the location
# of this script:
# 1. If the BASH_SOURCE variable is set, then use it.
# 2. Otherwise, if the _ variable is set, then use it.
# 3. Otherwise, use $0.
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "$_" ]; then
    SCRIPT_SOURCE="$_"
else
    SCRIPT_SOURCE="$0"
fi

# The following logic is used to determine the directory that contains
# this script:
# 1. If the SCRIPT_SOURCE variable is an absolute path, then use its
#    directory.
# 2. Otherwise, if the SCRIPT_SOURCE variable is a relative path, then
#    resolve it relative to the current working directory.
if [ "${SCRIPT_SOURCE#/}" != "$SCRIPT_SOURCE" ]; then
    SCRIPT_DIR="$(dirname "$SCRIPT_SOURCE")"
else
    SCRIPT_DIR="$(dirname "$(pwd)/$SCRIPT_SOURCE")"
fi

# The following logic is used to determine the location of the
# gradle-wrapper.jar file:
# 1. If a gradle-wrapper.jar file exists in the same directory as this
#    script, then use it.
# 2. Otherwise, if a gradle/wrapper/gradle-wrapper.jar file exists in the
#    directory that contains this script, then use it.
# 3. Otherwise, print an error message and exit.
if [ -f "$SCRIPT_DIR/gradle-wrapper.jar" ]; then
    WRAPPER_JAR="$SCRIPT_DIR/gradle-wrapper.jar"
elif [ -f "$SCRIPT_DIR/gradle/wrapper/gradle-wrapper.jar" ]; then
    WRAPPER_JAR="$SCRIPT_DIR/gradle/wrapper/gradle-wrapper.jar"
else
    echo "Error: gradle-wrapper.jar not found." >&2
    echo "This script is intended for use with the Gradle wrapper." >&2
    echo "Please see https://docs.gradle.org/current/userguide/gradle_wrapper.html for more information." >&2
    exit 1
fi

# Use the maximum available heap size for the Gradle daemon.
# This can be overridden by setting the GRADLE_OPTS environment variable.
if [ -z "$GRADLE_OPTS" ]; then
    GRADLE_OPTS="-Xmx64m -Xms64m"
fi

# If the JAVA_HOME environment variable is set, then use it.
# Otherwise, try to find java in the PATH.
if [ -n "$JAVA_HOME" ]; then
    JAVA_EXE="$JAVA_HOME/bin/java"
else
    JAVA_EXE="java"
fi

# Execute the gradle-wrapper.jar file.
exec "$JAVA_EXE" $GRADLE_OPTS -jar "$WRAPPER_JAR" "$@"
