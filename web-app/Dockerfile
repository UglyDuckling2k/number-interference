FROM node:18

# Set working directory
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Build the SvelteKit application
RUN npm run build

# Expose the port the app runs on
EXPOSE 4173

# Start the application
CMD ["npm","run", "preview","--", "--host", "0.0.0.0"]