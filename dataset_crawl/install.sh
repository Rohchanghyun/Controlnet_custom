# Firefox 및 의존성 설치
apt-get update && apt-get install -y firefox wget xvfb

# geckodriver 설치
wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz
tar -xzf geckodriver-v0.34.0-linux64.tar.gz
chmod +x geckodriver
mv geckodriver /usr/bin/
rm geckodriver-v0.34.0-linux64.tar.gz