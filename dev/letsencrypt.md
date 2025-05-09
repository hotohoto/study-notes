# Let's Encrypt



- https://letsencrypt.org/
- https://doubly8f.netlify.app/%EA%B0%9C%EB%B0%9C/2020/07/12/ssl-letsencrypt/



- ACME
  - https://en.wikipedia.org/wiki/Automatic_Certificate_Management_Environment
- Challenge types
  - https://letsencrypt.org/docs/challenge-types/
  - HTTP-01 challenge
    - challenges to put a file in your web server
  - DNS-01 challenge
    - challenges to change TXT record
      - automation only makes sense if your DNS provider has an API to do so
    - supports wildcard domain names (*.example.com)
