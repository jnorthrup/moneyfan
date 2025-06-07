@rem
@rem Copyright 2007-2022 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@rem This script is intended for use with the Gradle wrapper.
@rem It is a thin wrapper around the gradle-wrapper.jar file that ships with
@rem the Gradle wrapper.
@rem
@rem This script is not intended to be run directly. Instead, you should use
@rem the gradlew command in your project's root directory.
@rem
@rem For more information about the Gradle wrapper, please see
@rem https://docs.gradle.org/current/userguide/gradle_wrapper.html
@rem

@echo off

setlocal

rem Determine the location of this script.
set SCRIPT_DIR=%~dp0

rem The following logic is used to determine the location of the
rem gradle-wrapper.jar file:
rem 1. If a gradle-wrapper.jar file exists in the same directory as this
rem    script, then use it.
rem 2. Otherwise, if a gradle\wrapper\gradle-wrapper.jar file exists in the
rem    directory that contains this script, then use it.
rem 3. Otherwise, print an error message and exit.
if exist "%SCRIPT_DIR%gradle-wrapper.jar" (
    set WRAPPER_JAR="%SCRIPT_DIR%gradle-wrapper.jar"
) else if exist "%SCRIPT_DIR%gradle\wrapper\gradle-wrapper.jar" (
    set WRAPPER_JAR="%SCRIPT_DIR%gradle\wrapper\gradle-wrapper.jar"
) else (
    echo Error: gradle-wrapper.jar not found. >&2
    echo This script is intended for use with the Gradle wrapper. >&2
    echo Please see https://docs.gradle.org/current/userguide/gradle_wrapper.html for more information. >&2
    exit /B 1
)

rem Use the maximum available heap size for the Gradle daemon.
rem This can be overridden by setting the GRADLE_OPTS environment variable.
if "%GRADLE_OPTS%" == "" (
    set GRADLE_OPTS=-Xmx64m -Xms64m
)

rem If the JAVA_HOME environment variable is set, then use it.
rem Otherwise, try to find java in the PATH.
if not "%JAVA_HOME%" == "" (
    set JAVA_EXE="%JAVA_HOME%\bin\java.exe"
) else (
    set JAVA_EXE=java.exe
)

rem Execute the gradle-wrapper.jar file.
%JAVA_EXE% %GRADLE_OPTS% -jar %WRAPPER_JAR% %*

endlocal
