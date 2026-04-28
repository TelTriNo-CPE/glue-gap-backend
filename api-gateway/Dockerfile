# Stage 1: Build
FROM node:20-alpine AS builder

# Install pnpm
RUN npm install -g pnpm

WORKDIR /app

# Copy lockfile and package.json
COPY package.json pnpm-lock.yaml ./

# Install all dependencies
RUN pnpm install --frozen-lockfile

# Copy source code and config files
COPY nest-cli.json tsconfig.json tsconfig.build.json ./
COPY src ./src

# Build the application
RUN pnpm run build

# Remove development dependencies
RUN pnpm prune --prod

# Stage 2: Production
FROM node:20-alpine AS production

WORKDIR /app

# Create a non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy only necessary files from the build stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Use the non-root user
USER appuser

# Expose the requested port
EXPOSE 3000

# Set the command to start the app in production mode
CMD ["node", "dist/main"]
