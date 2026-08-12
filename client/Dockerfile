# Multi-stage build: compile the CRA app, then serve the static bundle with
# nginx. CRA bakes REACT_APP_* variables into the JS bundle at *build* time,
# so they're passed in as build args (see docker-compose.yml) rather than
# read from the environment at container start.
FROM node:20-alpine AS build
WORKDIR /app

ARG REACT_APP_SIGNALING_SERVER_URL=http://localhost:3001
ARG REACT_APP_PREDICTION_API_URL=http://localhost:5000
ARG REACT_APP_SELFTEST_API_URL=http://localhost:5001
ENV REACT_APP_SIGNALING_SERVER_URL=$REACT_APP_SIGNALING_SERVER_URL \
    REACT_APP_PREDICTION_API_URL=$REACT_APP_PREDICTION_API_URL \
    REACT_APP_SELFTEST_API_URL=$REACT_APP_SELFTEST_API_URL \
    CI=true

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
