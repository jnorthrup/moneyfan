"""
Kotlin Adapter
==============

Adapter to interface with coinbaseXChangeBot.main.kts via stdin/stdout.
"""

import json
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os


class KotlinAdapter:
    """
    Adapter for coinbaseXChangeBot.main.kts execution
    """
    
    def __init__(self, 
                 kotlin_script: str = "coinbaseXChangeBot.main.kts",
                 kotlin_path: str = None,
                 timeout: float = 30.0):
        """
        Initialize Kotlin adapter
        
        Args:
            kotlin_script: Path to Kotlin script
            kotlin_path: Path to Kotlin executable (if not in PATH)
            timeout: Timeout for Kotlin execution
        """
        self.kotlin_script = kotlin_script
        self.kotlin_path = kotlin_path or self._find_kotlin()
        self.timeout = timeout
        self.process = None
        
        # Validate Kotlin installation
        if not self.kotlin_path:
            raise RuntimeError(
                "Kotlin not found. Please install Kotlin or specify kotlin_path"
            )
        
        # Validate script exists
        if not os.path.exists(kotlin_script):
            raise FileNotFoundError(f"Kotlin script not found: {kotlin_script}")
        
        print(f"✅ Kotlin adapter initialized with: {self.kotlin_path}")
    
    def _find_kotlin(self) -> Optional[str]:
        """
        Find Kotlin executable in system PATH
        
        Returns:
            Path to Kotlin executable or None
        """
        # Try common Kotlin installation paths
        possible_paths = [
            'kotlin',  # In PATH
            '/usr/local/bin/kotlin',
            '/usr/bin/kotlin',
            os.path.expanduser('~/bin/kotlin'),
        ]
        
        for path in possible_paths:
            try:
                # Try to run kotlin --version
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                continue
        
        # Check if kotlin is available via jar
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return 'java'  # Will use kotlin.jar
        except:
            pass
        
        return None
    
    def start(self) -> bool:
        """
        Start Kotlin process
        
        Returns:
            True if started successfully
        """
        if self.process is not None:
            print("⚠️  Kotlin process already running")
            return False
        
        try:
            # Start Kotlin script with stdin/stdout pipes
            cmd = [self.kotlin_path, self.kotlin_script]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            print(f"✅ Started Kotlin process: {' '.join(cmd)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Kotlin process: {e}")
            return False
    
    def stop(self) -> None:
        """Stop Kotlin process"""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("✅ Stopped Kotlin process")
            except Exception as e:
                print(f"⚠️  Error stopping Kotlin process: {e}")
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
    
    def send_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send signal to Kotlin process via stdin
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if sent successfully
        """
        if self.process is None:
            print("⚠️  Kotlin process not running")
            return False
        
        try:
            # Convert to JSON
            json_signal = json.dumps(signal, ensure_ascii=False) + '\n'
            
            # Write to stdin
            self.process.stdin.write(json_signal)
            self.process.stdin.flush()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send signal to Kotlin: {e}")
            return False
    
    def read_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Read response from Kotlin process
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Response dictionary or None
        """
        if self.process is None:
            return None
        
        import select
        
        # Check if stdout is ready
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        
        if not readable:
            return None
        
        # Read line
        line = self.process.stdout.readline()
        
        if not line:
            return None
        
        line = line.strip()
        
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON from Kotlin: {e}")
            return None
    
    def send_and_receive(self, signal: Dict[str, Any], 
                        timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Send signal and receive response
        
        Args:
            signal: Signal to send
            timeout: Response timeout
            
        Returns:
            Response or None
        """
        # Send signal
        if not self.send_signal(signal):
            return None
        
        # Read response
        return self.read_response(timeout)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get Kotlin process status
        
        Returns:
            Status dictionary
        """
        status = {
            'process_running': self.process is not None,
            'kotlin_path': self.kotlin_path,
            'script': self.kotlin_script,
        }
        
        if self.process is not None:
            status['pid'] = self.process.pid
            status['returncode'] = self.process.poll()
        
        return status


class KotlinSignalExecutor:
    """
    Execute signals via Kotlin adapter
    """
    
    def __init__(self, adapter: KotlinAdapter):
        self.adapter = adapter
        self.signals_sent = 0
        self.responses_received = 0
        self.last_error = None
        
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal via Kotlin
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Execution result
        """
        try:
            # Add execution metadata
            signal['executed_at'] = datetime.now().isoformat()
            signal['execution_id'] = f"exec_{self.signals_sent}_{int(time.time())}"
            
            # Send and receive
            response = self.adapter.send_and_receive(signal, timeout=10.0)
            
            if response is None:
                result = {
                    'success': False,
                    'error': 'No response from Kotlin',
                    'signal_id': signal.get('execution_id'),
                    'timestamp': datetime.now().isoformat(),
                }
            else:
                self.responses_received += 1
                result = {
                    'success': response.get('success', False),
                    'response': response,
                    'signal_id': signal.get('execution_id'),
                    'timestamp': datetime.now().isoformat(),
                }
            
            self.signals_sent += 1
            
            return result
            
        except Exception as e:
            self.last_error = str(e)
            result = {
                'success': False,
                'error': str(e),
                'signal_id': signal.get('execution_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
            }
            return result
    
    def execute_batch(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute batch of signals
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            List of execution results
        """
        results = []
        
        for signal in signals:
            result = self.execute_signal(signal)
            results.append(result)
            
            # Small delay between signals
            time.sleep(0.1)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'signals_sent': self.signals_sent,
            'responses_received': self.responses_received,
            'success_rate': self.responses_received / max(1, self.signals_sent),
            'last_error': self.last_error,
        }


# Factory functions
def create_kotlin_adapter(script: str = "coinbaseXChangeBot.main.kts") -> KotlinAdapter:
    """
    Create Kotlin adapter for coinbaseXChangeBot.main.kts
    
    Args:
        script: Path to Kotlin script
        
    Returns:
        KotlinAdapter instance
    """
    return KotlinAdapter(script)


def create_signal_executor(adapter: KotlinAdapter) -> KotlinSignalExecutor:
    """
    Create signal executor
    
    Args:
        adapter: Kotlin adapter
        
    Returns:
        KotlinSignalExecutor instance
    """
    return KotlinSignalExecutor(adapter)